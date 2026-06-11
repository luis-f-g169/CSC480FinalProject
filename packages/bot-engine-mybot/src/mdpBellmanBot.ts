import {
    applyGameMove,
    BotEngineCapabilities,
    BotEngineInterface,
    BotEngineSuggestionResult,
    cloneGameState,
    GameState,
    getCellKey,
    getHexDistance,
    HexCoordinate,
    isCellWithinPlacementRadius,
} from "@ih3t/shared";

import learnedWeights from "./learnedWeights.json";

const WINNING_LINE_LENGTH = 6;
const DIRECTIONS: readonly HexCoordinate[] = [
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: 1, y: -1 },
];

const SEARCH_NEIGHBOR_RADIUS = 3;
const LINE_PROJECTION_RADIUS = 5;
const FIRST_MOVE_LIMIT = 14;
const SECOND_MOVE_LIMIT = 10;
const MAX_ACTIONS_PER_STATE = 64;
const TRANSPOSITION_LIMIT = 40_000;

type LineDetails = {
    length: number;
    openEnds: number;
};

type FeatureStats = {
    longest: number;
    openThree: number;
    openFour: number;
    openFive: number;
    closedFive: number;
    centerWeight: number;
};

type BellmanContext = {
    perspectivePlayerId: string;
    deadlineAt: number;
    nodes: number;
    truncated: boolean;
    transpositions: Map<string, number>;
};

type CandidateAction = {
    moves: HexCoordinate[];
    prior: number;
};

const DEFAULT_BACKUP_WEIGHTS = [
    -0.2424,
    1.7616,
    -1.3886,
    0.4377,
    -0.5778,
    1.2406,
    -1.0871,
    1.7273,
    -2.0764,
    0.8789,
    -1.1593,
    0.1334,
    -0.1603,
] as const;

/*
 * This bot is deliberately separate from MyBot Learning Alpha-Beta.
 *
 * It treats the game as a deterministic, two-player, zero-sum MDP:
 * - state: the current board
 * - action: a legal one- or two-placement turn
 * - transition: applyGameMove
 * - reward/value: +1 win, -1 loss, approximate value otherwise
 *
 * For tractable horizons it applies the adversarial Bellman backup:
 *     V(s) = max_a V(T(s,a)) on our turn
 *     V(s) = min_a V(T(s,a)) on the opponent turn
 *
 * When the infinite board/action space is too large for exact solving, it falls
 * back to the saved learned value function. That is the Approximate RL piece.
 */
class MdpBellmanBotEngine implements BotEngineInterface {
    private readonly backupWeights = getBackupWeights();

    getDisplayName(): string {
        return `MDP Bellman Bot`;
    }

    getCapabilities(): Readonly<BotEngineCapabilities> {
        return {
            suggestTurn: true,
            suggestMove: false,
        };
    }

    async suggestTurn(
        gameState: GameState,
        timeoutMs: number,
    ): Promise<BotEngineSuggestionResult<[HexCoordinate, HexCoordinate]>> {
        const playerId = gameState.currentTurnPlayerId;
        if (!playerId) {
            return { status: `failure`, message: `No current player to move.`, metadata: {} };
        }

        const deadlineAt = Date.now() + Math.max(50, timeoutMs - 20);
        const context: BellmanContext = {
            perspectivePlayerId: playerId,
            deadlineAt,
            nodes: 0,
            truncated: false,
            transpositions: new Map(),
        };
        const action = chooseBellmanAction(gameState, context, getBellmanDepth(timeoutMs), this.backupWeights);
        const suggestion = padAction(gameState, playerId, action?.moves ?? []);

        return {
            status: `provide`,
            suggestion,
            metadata: {
                mode: context.truncated ? `approximate-rl-backup` : `bellman-search`,
                nodes: String(context.nodes),
                transpositions: String(context.transpositions.size),
                depth: String(getBellmanDepth(timeoutMs)),
            },
        };
    }

    async suggestMove(
        _gameState: GameState,
        _timeoutMs: number,
    ): Promise<BotEngineSuggestionResult<HexCoordinate>> {
        return { status: `failure`, message: `MDP Bellman Bot searches full turns only.`, metadata: {} };
    }

    shutdown(): void {
        /* no resources to release */
    }
}

function chooseBellmanAction(
    gameState: GameState,
    context: BellmanContext,
    depth: number,
    backupWeights: readonly number[],
): CandidateAction | null {
    const playerId = gameState.currentTurnPlayerId;
    if (!playerId) {
        return null;
    }

    let bestAction: CandidateAction | null = null;
    let bestValue = Number.NEGATIVE_INFINITY;
    let alpha = Number.NEGATIVE_INFINITY;

    for (const action of generateCandidateActions(gameState, playerId)) {
        if (isPastDeadline(context)) {
            break;
        }

        const nextState = applyAction(gameState, playerId, action.moves);
        const value = bellmanValue(nextState, depth - 1, alpha, Number.POSITIVE_INFINITY, context, backupWeights)
            + action.prior * 0.00001;

        if (value > bestValue) {
            bestValue = value;
            bestAction = action;
        }

        alpha = Math.max(alpha, bestValue);
    }

    return bestAction;
}

function bellmanValue(
    gameState: GameState,
    depth: number,
    alpha: number,
    beta: number,
    context: BellmanContext,
    backupWeights: readonly number[],
): number {
    context.nodes += 1;

    const terminalValue = getTerminalValue(gameState, context.perspectivePlayerId);
    if (terminalValue !== null) {
        return terminalValue;
    }

    if (
        depth <= 0
        || !gameState.currentTurnPlayerId
        || isPastDeadline(context)
        || context.transpositions.size >= TRANSPOSITION_LIMIT
    ) {
        context.truncated = true;
        return approximateValue(gameState, context.perspectivePlayerId, backupWeights);
    }

    const cacheKey = createStateKey(gameState, depth, context.perspectivePlayerId);
    const cachedValue = context.transpositions.get(cacheKey);
    if (cachedValue !== undefined) {
        return cachedValue;
    }

    const currentPlayerId = gameState.currentTurnPlayerId;
    const maximizing = currentPlayerId === context.perspectivePlayerId;
    const actions = generateCandidateActions(gameState, currentPlayerId);
    if (actions.length === 0) {
        return approximateValue(gameState, context.perspectivePlayerId, backupWeights);
    }

    let bestValue = maximizing ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;

    for (const action of actions) {
        if (isPastDeadline(context)) {
            context.truncated = true;
            break;
        }

        const nextState = applyAction(gameState, currentPlayerId, action.moves);
        const value = bellmanValue(nextState, depth - 1, alpha, beta, context, backupWeights);

        if (maximizing) {
            bestValue = Math.max(bestValue, value);
            alpha = Math.max(alpha, bestValue);
        } else {
            bestValue = Math.min(bestValue, value);
            beta = Math.min(beta, bestValue);
        }

        if (beta <= alpha) {
            break;
        }
    }

    const resolvedValue = Number.isFinite(bestValue)
        ? bestValue
        : approximateValue(gameState, context.perspectivePlayerId, backupWeights);
    context.transpositions.set(cacheKey, resolvedValue);
    return resolvedValue;
}

function generateCandidateActions(gameState: GameState, playerId: string): CandidateAction[] {
    const firstMoves = scoreLegalMoves(gameState, playerId, FIRST_MOVE_LIMIT);
    const actions: CandidateAction[] = [];

    for (const firstMove of firstMoves) {
        const afterFirstMove = applyAction(gameState, playerId, [firstMove.move]);
        if (afterFirstMove.winner || afterFirstMove.currentTurnPlayerId !== playerId) {
            actions.push({ moves: [firstMove.move], prior: firstMove.score + 1_000_000 });
            continue;
        }

        for (const secondMove of scoreLegalMoves(afterFirstMove, playerId, SECOND_MOVE_LIMIT)) {
            actions.push({
                moves: [firstMove.move, secondMove.move],
                prior: firstMove.score + secondMove.score,
            });
        }
    }

    return dedupeActions(actions)
        .sort((a, b) => b.prior - a.prior)
        .slice(0, MAX_ACTIONS_PER_STATE);
}

function scoreLegalMoves(gameState: GameState, playerId: string, limit: number) {
    return getCandidateCells(gameState)
        .map(move => ({
            move,
            score: scoreMove(gameState, playerId, move),
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
}

function getCandidateCells(gameState: GameState): HexCoordinate[] {
    const occupied = new Set(gameState.cells.map(cell => getCellKey(cell.x, cell.y)));
    if (gameState.cells.length === 0) {
        return [{ x: 0, y: 0 }];
    }

    const candidates = new Map<string, HexCoordinate>();

    for (const anchor of gameState.cells) {
        for (let dx = -SEARCH_NEIGHBOR_RADIUS; dx <= SEARCH_NEIGHBOR_RADIUS; dx++) {
            for (let dy = -SEARCH_NEIGHBOR_RADIUS; dy <= SEARCH_NEIGHBOR_RADIUS; dy++) {
                const candidate = { x: anchor.x + dx, y: anchor.y + dy };
                if (getHexDistance(anchor, candidate) <= SEARCH_NEIGHBOR_RADIUS) {
                    addCandidate(candidates, occupied, gameState, candidate);
                }
            }
        }

        for (const direction of DIRECTIONS) {
            for (let step = -LINE_PROJECTION_RADIUS; step <= LINE_PROJECTION_RADIUS; step++) {
                if (step !== 0) {
                    addCandidate(candidates, occupied, gameState, {
                        x: anchor.x + direction.x * step,
                        y: anchor.y + direction.y * step,
                    });
                }
            }
        }
    }

    return [...candidates.values()];
}

function addCandidate(
    candidates: Map<string, HexCoordinate>,
    occupied: ReadonlySet<string>,
    gameState: GameState,
    candidate: HexCoordinate,
): void {
    const key = getCellKey(candidate.x, candidate.y);
    if (
        !occupied.has(key)
        && isCellWithinPlacementRadius(gameState.cells, candidate)
    ) {
        candidates.set(key, candidate);
    }
}

function scoreMove(gameState: GameState, playerId: string, move: HexCoordinate): number {
    const opponentId = getOpponentPlayerId(gameState, playerId);
    const occupied = new Set(gameState.cells.map(cell => getCellKey(cell.x, cell.y)));
    const ownLine = getBestLineDetailsThroughMove(getPlayerCellSet(gameState, playerId), occupied, move);
    const opponentLine = opponentId
        ? getBestLineDetailsThroughMove(getPlayerCellSet(gameState, opponentId), occupied, move)
        : { length: 1, openEnds: 0 };

    if (ownLine.length >= WINNING_LINE_LENGTH) {
        return 1_000_000;
    }

    if (opponentLine.length >= WINNING_LINE_LENGTH) {
        return 900_000;
    }

    return scoreLineDetails(ownLine, 1)
        + scoreLineDetails(opponentLine, 1.2)
        - Math.abs(move.x) * 0.03
        - Math.abs(move.y) * 0.03
        - Math.abs(move.x + move.y) * 0.02;
}

function approximateValue(
    gameState: GameState,
    playerId: string,
    backupWeights: readonly number[],
): number {
    const terminalValue = getTerminalValue(gameState, playerId);
    if (terminalValue !== null) {
        return terminalValue;
    }

    const learnedValue = Math.tanh(dot(backupWeights, extractFeatures(gameState, playerId)));
    const tacticalValue = tacticalEvaluation(gameState, playerId);
    return clamp((learnedValue * 0.55) + (tacticalValue * 0.45));
}

function extractFeatures(gameState: GameState, playerId: string): number[] {
    const opponentId = getOpponentPlayerId(gameState, playerId);
    const ownStats = collectFeatureStats(gameState, playerId);
    const opponentStats = opponentId ? collectFeatureStats(gameState, opponentId) : createEmptyFeatureStats();

    return [
        1,
        ownStats.longest / WINNING_LINE_LENGTH,
        opponentStats.longest / WINNING_LINE_LENGTH,
        ownStats.openThree / 8,
        opponentStats.openThree / 8,
        ownStats.openFour / 6,
        opponentStats.openFour / 6,
        ownStats.openFive / 4,
        opponentStats.openFive / 4,
        ownStats.closedFive / 4,
        opponentStats.closedFive / 4,
        ownStats.centerWeight / 12,
        opponentStats.centerWeight / 12,
    ];
}

function tacticalEvaluation(gameState: GameState, playerId: string): number {
    const opponentId = getOpponentPlayerId(gameState, playerId);
    const ownStats = collectFeatureStats(gameState, playerId);
    const opponentStats = opponentId ? collectFeatureStats(gameState, opponentId) : createEmptyFeatureStats();
    const score = (ownStats.longest - opponentStats.longest) * 0.35
        + ownStats.openThree * 0.08
        - opponentStats.openThree * 0.1
        + ownStats.openFour * 0.22
        - opponentStats.openFour * 0.32
        + ownStats.openFive * 0.9
        - opponentStats.openFive * 1.15
        + ownStats.closedFive * 0.3
        - opponentStats.closedFive * 0.38
        + (ownStats.centerWeight - opponentStats.centerWeight) * 0.04;

    return Math.tanh(score);
}

function collectFeatureStats(gameState: GameState, playerId: string): FeatureStats {
    const playerCells = gameState.cells.filter(cell => cell.occupiedBy === playerId);
    const playerSet = new Set(playerCells.map(cell => getCellKey(cell.x, cell.y)));
    const occupied = new Set(gameState.cells.map(cell => getCellKey(cell.x, cell.y)));
    const stats = createEmptyFeatureStats();

    for (const cell of playerCells) {
        stats.centerWeight += 1 / (1 + getHexDistance(cell, { x: 0, y: 0 }));

        for (const direction of DIRECTIONS) {
            if (playerSet.has(getCellKey(cell.x - direction.x, cell.y - direction.y))) {
                continue;
            }

            const line = collectLineFromStart(playerSet, cell, direction);
            const openEnds = countOpenEnds(occupied, line[0], line[line.length - 1], direction);
            stats.longest = Math.max(stats.longest, line.length);

            if (line.length === 3 && openEnds > 0) {
                stats.openThree += openEnds;
            } else if (line.length === 4 && openEnds > 0) {
                stats.openFour += openEnds;
            } else if (line.length === 5) {
                if (openEnds > 0) {
                    stats.openFive += openEnds;
                } else {
                    stats.closedFive += 1;
                }
            }
        }
    }

    return stats;
}

function getBestLineDetailsThroughMove(
    playerSet: ReadonlySet<string>,
    occupied: ReadonlySet<string>,
    move: HexCoordinate,
): LineDetails {
    let bestDetails: LineDetails = { length: 1, openEnds: 0 };

    for (const direction of DIRECTIONS) {
        const forward = countConnected(playerSet, move, direction.x, direction.y);
        const backward = countConnected(playerSet, move, -direction.x, -direction.y);
        const length = 1 + forward + backward;
        const first = {
            x: move.x - direction.x * backward,
            y: move.y - direction.y * backward,
        };
        const last = {
            x: move.x + direction.x * forward,
            y: move.y + direction.y * forward,
        };
        const openEnds = countOpenEnds(occupied, first, last, direction);

        if (length > bestDetails.length || (length === bestDetails.length && openEnds > bestDetails.openEnds)) {
            bestDetails = { length, openEnds };
        }
    }

    return bestDetails;
}

function scoreLineDetails(details: LineDetails, urgencyMultiplier: number): number {
    if (details.length >= 5) {
        return (80_000 + details.openEnds * 12_000) * urgencyMultiplier;
    }

    if (details.length === 4) {
        return (9_000 + details.openEnds * 2_000) * urgencyMultiplier;
    }

    if (details.length === 3) {
        return (900 + details.openEnds * 240) * urgencyMultiplier;
    }

    return details.length * details.length * 10 * urgencyMultiplier;
}

function collectLineFromStart(
    playerSet: ReadonlySet<string>,
    start: HexCoordinate,
    direction: HexCoordinate,
): HexCoordinate[] {
    const line = [start];
    let current = { x: start.x + direction.x, y: start.y + direction.y };

    while (playerSet.has(getCellKey(current.x, current.y))) {
        line.push(current);
        current = { x: current.x + direction.x, y: current.y + direction.y };
    }

    return line;
}

function countConnected(
    playerSet: ReadonlySet<string>,
    start: HexCoordinate,
    directionX: number,
    directionY: number,
): number {
    let total = 0;
    let current = { x: start.x + directionX, y: start.y + directionY };

    while (playerSet.has(getCellKey(current.x, current.y))) {
        total += 1;
        current = { x: current.x + directionX, y: current.y + directionY };
    }

    return total;
}

function countOpenEnds(
    occupied: ReadonlySet<string>,
    first: HexCoordinate,
    last: HexCoordinate,
    direction: HexCoordinate,
): number {
    let openEnds = 0;
    const before = { x: first.x - direction.x, y: first.y - direction.y };
    const after = { x: last.x + direction.x, y: last.y + direction.y };

    if (!occupied.has(getCellKey(before.x, before.y))) {
        openEnds += 1;
    }

    if (!occupied.has(getCellKey(after.x, after.y))) {
        openEnds += 1;
    }

    return openEnds;
}

function applyAction(gameState: GameState, playerId: string, moves: readonly HexCoordinate[]): GameState {
    const nextState = cloneGameState(gameState);

    for (const move of moves) {
        if (nextState.winner || nextState.currentTurnPlayerId !== playerId) {
            break;
        }

        applyGameMove(nextState, { playerId, x: move.x, y: move.y });
    }

    return nextState;
}

function padAction(
    gameState: GameState,
    playerId: string,
    moves: readonly HexCoordinate[],
): [HexCoordinate, HexCoordinate] {
    const paddedMoves = [...moves];
    let nextState = cloneGameState(gameState);

    for (const move of paddedMoves) {
        if (nextState.winner || nextState.currentTurnPlayerId !== playerId) {
            break;
        }

        applyGameMove(nextState, { playerId, x: move.x, y: move.y });
    }

    while (paddedMoves.length < 2) {
        const fallback = scoreLegalMoves(nextState, playerId, 1)[0]?.move ?? paddedMoves[0] ?? { x: 0, y: 0 };
        paddedMoves.push(fallback);
        if (!nextState.winner && nextState.currentTurnPlayerId === playerId) {
            nextState = applyAction(nextState, playerId, [fallback]);
        }
    }

    return [paddedMoves[0], paddedMoves[1]];
}

function dedupeActions(actions: readonly CandidateAction[]): CandidateAction[] {
    const seen = new Set<string>();
    const dedupedActions: CandidateAction[] = [];

    for (const action of actions) {
        const key = action.moves
            .map(move => getCellKey(move.x, move.y))
            .sort()
            .join(`|`);
        if (!seen.has(key)) {
            seen.add(key);
            dedupedActions.push(action);
        }
    }

    return dedupedActions;
}

function createStateKey(gameState: GameState, depth: number, perspectivePlayerId: string): string {
    const cells = gameState.cells
        .map(cell => `${cell.x},${cell.y},${cell.occupiedBy === perspectivePlayerId ? `p` : `o`}`)
        .sort()
        .join(`;`);

    return `${depth}:${gameState.currentTurnPlayerId === perspectivePlayerId ? `p` : `o`}:${gameState.placementsRemaining}:${cells}`;
}

function getPlayerCellSet(gameState: GameState, playerId: string): Set<string> {
    return new Set(
        gameState.cells
            .filter(cell => cell.occupiedBy === playerId)
            .map(cell => getCellKey(cell.x, cell.y)),
    );
}

function getOpponentPlayerId(gameState: GameState, playerId: string): string | null {
    return Object.keys(gameState.playerTiles).find(existingPlayerId => existingPlayerId !== playerId) ?? null;
}

function getTerminalValue(gameState: GameState, perspectivePlayerId: string): number | null {
    if (!gameState.winner) {
        return null;
    }

    return gameState.winner.playerId === perspectivePlayerId ? 1 : -1;
}

function createEmptyFeatureStats(): FeatureStats {
    return {
        longest: 0,
        openThree: 0,
        openFour: 0,
        openFive: 0,
        closedFive: 0,
        centerWeight: 0,
    };
}

function getBellmanDepth(timeoutMs: number): number {
    if (timeoutMs >= 15_000) {
        return 5;
    }

    if (timeoutMs >= 5_000) {
        return 4;
    }

    if (timeoutMs >= 2_000) {
        return 3;
    }

    return 2;
}

function getBackupWeights(): readonly number[] {
    const weights = learnedWeights.weights;
    if (
        Array.isArray(weights)
        && weights.length === DEFAULT_BACKUP_WEIGHTS.length
        && weights.every(weight => Number.isFinite(weight))
    ) {
        return weights;
    }

    return DEFAULT_BACKUP_WEIGHTS;
}

function isPastDeadline(context: BellmanContext): boolean {
    if (Date.now() < context.deadlineAt) {
        return false;
    }

    context.truncated = true;
    return true;
}

function dot(weights: readonly number[], features: readonly number[]): number {
    return weights.reduce((total, weight, index) => total + weight * (features[index] ?? 0), 0);
}

function clamp(value: number): number {
    return Math.max(-1, Math.min(1, value));
}

export default async function () {
    return new MdpBellmanBotEngine();
}
