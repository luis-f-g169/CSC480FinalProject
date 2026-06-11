import {
    applyGameMove,
    type BotEngineCapabilities,
    type BotEngineInterface,
    type BotEngineSuggestionResult,
    cloneGameState,
    type GameState,
    getCellKey,
    getHexDistance,
    type HexCoordinate,
    isCellWithinPlacementRadius,
} from "@ih3t/shared";

const WIN_SCORE = 10_000_000;
const WINNING_LINE_LENGTH = 6;
const MAX_SEARCH_DEPTH = 4;
const ROOT_ACTION_LIMIT = 90;
const INNER_ACTION_LIMIT = 44;
const ROOT_FIRST_MOVE_LIMIT = 26;
const INNER_FIRST_MOVE_LIMIT = 16;
const SECOND_MOVE_LIMIT = 14;
const RESPONSE_TIME_RATIO = 0.88;
const MIN_SEARCH_TIME_MS = 35;
const SEARCH_SAFETY_MS = 8;

const DIRECTIONS = [
    [1, 0],
    [0, 1],
    [1, -1],
] as const;

const LINE_WEIGHTS = [
    0,
    3,
    30,
    280,
    2_700,
    95_000,
    WIN_SCORE,
] as const;

type Direction = typeof DIRECTIONS[number];

type ScoredMove = {
    move: HexCoordinate;
    score: number;
};

type TurnAction = {
    moves: HexCoordinate[];
    scoreHint: number;
};

type LineWindow = {
    emptyCells: HexCoordinate[];
    playerCount: number;
};

type SearchContext = {
    rootPlayerId: string;
    deadline: number;
    nodes: number;
};

class SearchTimeout extends Error {
    constructor() {
        super(`Search timed out.`);
        this.name = `SearchTimeout`;
    }
}

class MyBotEngine implements BotEngineInterface {
    getDisplayName(): string {
        return `MyBot AlphaBeta`;
    }

    getCapabilities(): Readonly<BotEngineCapabilities> {
        return {
            suggestTurn: true,
            suggestMove: true,
        };
    }

    async suggestTurn(
        gameState: GameState,
        timeoutMs: number,
    ): Promise<BotEngineSuggestionResult<[HexCoordinate, HexCoordinate]>> {
        const playerId = gameState.currentTurnPlayerId;
        if (!playerId) {
            return {
                status: `failure`,
                message: `No active player to move.`,
                metadata: {},
            };
        }

        const startedAt = Date.now();
        const result = searchBestAction(gameState, playerId, timeoutMs);
        const firstMove = result.moves[0] ?? getBestSingleMove(gameState, playerId);
        const afterFirstMove = applyTurnAction(gameState, {
            moves: [firstMove],
            scoreHint: 0,
        });
        const secondMove = result.moves[1]
            ?? (afterFirstMove?.currentTurnPlayerId === playerId
                ? getBestSingleMove(afterFirstMove, playerId)
                : firstMove);

        return {
            status: `provide`,
            suggestion: [firstMove, secondMove],
            metadata: {
                depth: `${result.depth}`,
                nodes: `${result.nodes}`,
                elapsedMs: `${Date.now() - startedAt}`,
                score: `${Math.round(result.score)}`,
            },
        };
    }

    async suggestMove(
        gameState: GameState,
        timeoutMs: number,
    ): Promise<BotEngineSuggestionResult<HexCoordinate>> {
        const playerId = gameState.currentTurnPlayerId;
        if (!playerId) {
            return {
                status: `failure`,
                message: `No active player to move.`,
                metadata: {},
            };
        }

        const startedAt = Date.now();
        const result = searchBestAction(gameState, playerId, timeoutMs);
        return {
            status: `provide`,
            suggestion: result.moves[0] ?? getBestSingleMove(gameState, playerId),
            metadata: {
                depth: `${result.depth}`,
                nodes: `${result.nodes}`,
                elapsedMs: `${Date.now() - startedAt}`,
                score: `${Math.round(result.score)}`,
            },
        };
    }

    shutdown(): void {
        /* no external resources */
    }
}

function searchBestAction(gameState: GameState, playerId: string, timeoutMs: number) {
    if (gameState.cells.length === 0) {
        return {
            moves: [{ x: 0, y: 0 }],
            score: WIN_SCORE,
            depth: 0,
            nodes: 0,
        };
    }

    const searchTimeMs = Math.max(
        MIN_SEARCH_TIME_MS,
        Math.floor(timeoutMs * RESPONSE_TIME_RATIO) - SEARCH_SAFETY_MS,
    );
    const context: SearchContext = {
        rootPlayerId: playerId,
        deadline: Date.now() + searchTimeMs,
        nodes: 0,
    };
    let bestAction = getGreedyAction(gameState, playerId);
    let bestScore = Number.NEGATIVE_INFINITY;
    let completedDepth = 0;

    for (let depth = 1; depth <= MAX_SEARCH_DEPTH; depth += 1) {
        try {
            const result = searchRoot(gameState, depth, context);
            bestAction = result.action;
            bestScore = result.score;
            completedDepth = depth;
        } catch (error) {
            if (error instanceof SearchTimeout) {
                break;
            }

            throw error;
        }
    }

    if (bestScore === Number.NEGATIVE_INFINITY) {
        const nextState = applyTurnAction(gameState, bestAction);
        bestScore = nextState ? evaluatePosition(nextState, playerId) : evaluatePosition(gameState, playerId);
    }

    return {
        moves: bestAction.moves,
        score: bestScore,
        depth: completedDepth,
        nodes: context.nodes,
    };
}

function searchRoot(gameState: GameState, depth: number, context: SearchContext): {
    action: TurnAction;
    score: number;
} {
    assertTimeRemaining(context);

    const playerId = gameState.currentTurnPlayerId;
    if (!playerId) {
        return {
            action: getGreedyAction(gameState, context.rootPlayerId),
            score: evaluatePosition(gameState, context.rootPlayerId),
        };
    }

    const actions = generateTurnActions(gameState, playerId, {
        actionLimit: ROOT_ACTION_LIMIT,
        firstMoveLimit: ROOT_FIRST_MOVE_LIMIT,
    });
    if (actions.length === 0) {
        return {
            action: getGreedyAction(gameState, playerId),
            score: evaluatePosition(gameState, context.rootPlayerId),
        };
    }

    let alpha = Number.NEGATIVE_INFINITY;
    let bestScore = Number.NEGATIVE_INFINITY;
    let bestAction = actions[0];

    for (const action of actions) {
        assertTimeRemaining(context);
        const nextState = applyTurnAction(gameState, action);
        if (!nextState) {
            continue;
        }

        const score = alphaBeta(
            nextState,
            depth - 1,
            alpha,
            Number.POSITIVE_INFINITY,
            context,
        );
        if (score > bestScore) {
            bestScore = score;
            bestAction = action;
        }

        alpha = Math.max(alpha, bestScore);
    }

    return {
        action: bestAction,
        score: bestScore,
    };
}

function alphaBeta(
    gameState: GameState,
    depth: number,
    alpha: number,
    beta: number,
    context: SearchContext,
): number {
    context.nodes += 1;
    if (context.nodes % 64 === 0) {
        assertTimeRemaining(context);
    }

    if (gameState.winner) {
        return gameState.winner.playerId === context.rootPlayerId
            ? WIN_SCORE + depth
            : -WIN_SCORE - depth;
    }

    const playerId = gameState.currentTurnPlayerId;
    if (depth <= 0 || !playerId) {
        return evaluatePosition(gameState, context.rootPlayerId);
    }

    const actions = generateTurnActions(gameState, playerId, {
        actionLimit: INNER_ACTION_LIMIT,
        firstMoveLimit: INNER_FIRST_MOVE_LIMIT,
    });
    if (actions.length === 0) {
        return evaluatePosition(gameState, context.rootPlayerId);
    }

    const maximizing = playerId === context.rootPlayerId;
    if (maximizing) {
        let value = Number.NEGATIVE_INFINITY;
        for (const action of actions) {
            const nextState = applyTurnAction(gameState, action);
            if (!nextState) {
                continue;
            }

            value = Math.max(value, alphaBeta(nextState, depth - 1, alpha, beta, context));
            alpha = Math.max(alpha, value);
            if (alpha >= beta) {
                break;
            }
        }
        return value;
    }

    let value = Number.POSITIVE_INFINITY;
    for (const action of actions) {
        const nextState = applyTurnAction(gameState, action);
        if (!nextState) {
            continue;
        }

        value = Math.min(value, alphaBeta(nextState, depth - 1, alpha, beta, context));
        beta = Math.min(beta, value);
        if (alpha >= beta) {
            break;
        }
    }
    return value;
}

function generateTurnActions(
    gameState: GameState,
    playerId: string,
    options: {
        actionLimit: number;
        firstMoveLimit: number;
    },
): TurnAction[] {
    const winningAction = findImmediateWinningAction(gameState, playerId);
    if (winningAction) {
        return [winningAction];
    }

    const firstMoves = getScoredCandidateMoves(gameState, playerId)
        .slice(0, options.firstMoveLimit);
    const actions: TurnAction[] = [];

    for (const firstMove of firstMoves) {
        const afterFirstMove = applyMoveToClone(gameState, playerId, firstMove.move);
        if (!afterFirstMove) {
            continue;
        }

        if (
            afterFirstMove.winner
            || afterFirstMove.currentTurnPlayerId !== playerId
            || afterFirstMove.placementsRemaining <= 0
        ) {
            actions.push({
                moves: [firstMove.move],
                scoreHint: firstMove.score + evaluatePosition(afterFirstMove, playerId),
            });
            continue;
        }

        const secondMoves = getScoredCandidateMoves(afterFirstMove, playerId)
            .slice(0, SECOND_MOVE_LIMIT);
        for (const secondMove of secondMoves) {
            const afterSecondMove = applyMoveToClone(afterFirstMove, playerId, secondMove.move);
            if (!afterSecondMove) {
                continue;
            }

            actions.push({
                moves: [
                    firstMove.move,
                    secondMove.move,
                ],
                scoreHint: firstMove.score
                    + secondMove.score
                    + evaluatePosition(afterSecondMove, playerId) * 0.08,
            });
        }
    }

    if (actions.length === 0) {
        const fallbackMove = getBestSingleMove(gameState, playerId);
        actions.push({
            moves: [fallbackMove],
            scoreHint: scoreMove(gameState, playerId, fallbackMove),
        });
    }

    return actions
        .sort((left, right) => right.scoreHint - left.scoreHint)
        .slice(0, options.actionLimit);
}

function findImmediateWinningAction(gameState: GameState, playerId: string): TurnAction | null {
    const threats = collectThreats(gameState, playerId, 2)
        .sort((left, right) => left.emptyCells.length - right.emptyCells.length);
    const winningThreat = threats.find((threat) => threat.emptyCells.every((cell) => isLegalMove(gameState, cell)));
    if (!winningThreat) {
        return null;
    }

    return {
        moves: winningThreat.emptyCells,
        scoreHint: WIN_SCORE,
    };
}

function getGreedyAction(gameState: GameState, playerId: string): TurnAction {
    const firstMove = getBestSingleMove(gameState, playerId);
    const afterFirstMove = applyMoveToClone(gameState, playerId, firstMove);
    if (
        !afterFirstMove
        || afterFirstMove.winner
        || afterFirstMove.currentTurnPlayerId !== playerId
        || afterFirstMove.placementsRemaining <= 0
    ) {
        return {
            moves: [firstMove],
            scoreHint: scoreMove(gameState, playerId, firstMove),
        };
    }

    const secondMove = getBestSingleMove(afterFirstMove, playerId);
    return {
        moves: [
            firstMove,
            secondMove,
        ],
        scoreHint: scoreMove(gameState, playerId, firstMove)
            + scoreMove(afterFirstMove, playerId, secondMove),
    };
}

function getBestSingleMove(gameState: GameState, playerId: string): HexCoordinate {
    return getScoredCandidateMoves(gameState, playerId)[0]?.move ?? { x: 0, y: 0 };
}

function getScoredCandidateMoves(gameState: GameState, playerId: string): ScoredMove[] {
    return collectCandidateMoves(gameState, playerId)
        .map((move) => ({
            move,
            score: scoreMove(gameState, playerId, move),
        }))
        .sort((left, right) => right.score - left.score);
}

function collectCandidateMoves(gameState: GameState, playerId: string): HexCoordinate[] {
    const candidates = new Map<string, HexCoordinate>();
    const opponentId = getOpponentId(gameState, playerId);
    const addCandidate = (cell: HexCoordinate) => {
        if (!isLegalMove(gameState, cell)) {
            return;
        }

        candidates.set(getCellKey(cell.x, cell.y), cell);
    };

    if (gameState.cells.length === 0) {
        addCandidate({ x: 0, y: 0 });
        return [...candidates.values()];
    }

    for (const threat of collectThreats(gameState, playerId, 3)) {
        for (const cell of threat.emptyCells) {
            addCandidate(cell);
        }
    }

    if (opponentId) {
        for (const threat of collectThreats(gameState, opponentId, 3)) {
            for (const cell of threat.emptyCells) {
                addCandidate(cell);
            }
        }
    }

    for (const cell of gameState.cells) {
        for (let x = cell.x - 2; x <= cell.x + 2; x += 1) {
            for (let y = cell.y - 2; y <= cell.y + 2; y += 1) {
                const candidate = { x, y };
                if (getHexDistance(cell, candidate) <= 2) {
                    addCandidate(candidate);
                }
            }
        }

        for (const [directionX, directionY] of DIRECTIONS) {
            addCandidate({
                x: cell.x + directionX,
                y: cell.y + directionY,
            });
            addCandidate({
                x: cell.x - directionX,
                y: cell.y - directionY,
            });
        }
    }

    if (candidates.size < 10) {
        for (const cell of gameState.cells) {
            for (let x = cell.x - 3; x <= cell.x + 3; x += 1) {
                for (let y = cell.y - 3; y <= cell.y + 3; y += 1) {
                    const candidate = { x, y };
                    if (getHexDistance(cell, candidate) <= 3) {
                        addCandidate(candidate);
                    }
                }
            }
        }
    }

    return [...candidates.values()];
}

function scoreMove(gameState: GameState, playerId: string, move: HexCoordinate): number {
    if (!isLegalMove(gameState, move)) {
        return Number.NEGATIVE_INFINITY;
    }

    const nextState = applyMoveToClone(gameState, playerId, move);
    if (!nextState) {
        return Number.NEGATIVE_INFINITY;
    }

    if (nextState.winner?.playerId === playerId) {
        return WIN_SCORE;
    }

    const opponentId = getOpponentId(gameState, playerId);
    let score = evaluateMoveShape(gameState, playerId, move);
    score += countAdjacentCells(gameState, playerId, move, 1) * 85;
    score += countAdjacentCells(gameState, playerId, move, 2) * 22;
    score += countAdjacentCells(gameState, opponentId, move, 1) * 32;
    score -= getHexDistance({ x: 0, y: 0 }, move) * 0.35;

    if (opponentId) {
        for (const threat of collectThreats(gameState, opponentId, 2)) {
            if (threat.emptyCells.some((cell) => sameCell(cell, move))) {
                score += threat.emptyCells.length === 1 ? 180_000 : 38_000;
            }
        }
    }

    for (const threat of collectThreats(gameState, playerId, 2)) {
        if (threat.emptyCells.some((cell) => sameCell(cell, move))) {
            score += threat.emptyCells.length === 1 ? 260_000 : 55_000;
        }
    }

    return score;
}

function evaluatePosition(gameState: GameState, playerId: string): number {
    if (gameState.winner) {
        return gameState.winner.playerId === playerId ? WIN_SCORE : -WIN_SCORE;
    }

    const opponentId = getOpponentId(gameState, playerId);
    const ownLineScore = evaluateLinePotential(gameState, playerId);
    const opponentLineScore = opponentId ? evaluateLinePotential(gameState, opponentId) : 0;
    const ownThreatScore = evaluateThreats(gameState, playerId);
    const opponentThreatScore = opponentId ? evaluateThreats(gameState, opponentId) : 0;
    const tempoScore = gameState.currentTurnPlayerId === playerId ? 45 : -45;

    return ownLineScore
        - opponentLineScore * 1.18
        + ownThreatScore
        - opponentThreatScore * 1.35
        + tempoScore;
}

function evaluateLinePotential(gameState: GameState, playerId: string): number {
    let score = 0;
    for (const window of collectLineWindows(gameState, playerId)) {
        const emptyCount = window.emptyCells.length;
        const base = LINE_WEIGHTS[window.playerCount] ?? 0;
        score += base;

        if (window.playerCount >= 3) {
            score += Math.max(0, 4 - emptyCount) * 40;
        }
    }

    return score;
}

function evaluateThreats(gameState: GameState, playerId: string): number {
    return collectThreats(gameState, playerId, 2).reduce((score, threat) => {
        if (threat.emptyCells.length === 1) {
            return score + 120_000;
        }

        return score + 32_000;
    }, 0);
}

function evaluateMoveShape(gameState: GameState, playerId: string, move: HexCoordinate): number {
    const occupiedByCell = buildOccupiedMap(gameState);
    occupiedByCell.set(getCellKey(move.x, move.y), playerId);

    let score = 0;
    for (const [directionX, directionY] of DIRECTIONS) {
        for (let startOffset = -(WINNING_LINE_LENGTH - 1); startOffset <= 0; startOffset += 1) {
            let ownCount = 0;
            let blocked = false;
            for (let step = 0; step < WINNING_LINE_LENGTH; step += 1) {
                const x = move.x + directionX * (startOffset + step);
                const y = move.y + directionY * (startOffset + step);
                const occupant = occupiedByCell.get(getCellKey(x, y));
                if (!occupant) {
                    continue;
                }

                if (occupant === playerId) {
                    ownCount += 1;
                } else {
                    blocked = true;
                    break;
                }
            }

            if (!blocked) {
                score += LINE_WEIGHTS[ownCount] ?? 0;
            }
        }
    }

    return score;
}

function collectThreats(gameState: GameState, playerId: string, maxEmptyCells: number): LineWindow[] {
    return collectLineWindows(gameState, playerId)
        .filter((window) =>
            window.emptyCells.length > 0
            && window.emptyCells.length <= maxEmptyCells
            && window.playerCount + window.emptyCells.length === WINNING_LINE_LENGTH);
}

function collectLineWindows(gameState: GameState, playerId: string): LineWindow[] {
    const occupiedByCell = buildOccupiedMap(gameState);
    const playerCells = gameState.cells.filter((cell) => cell.occupiedBy === playerId);
    const windows = new Map<string, LineWindow>();

    for (const anchor of playerCells) {
        for (const direction of DIRECTIONS) {
            for (let startOffset = -(WINNING_LINE_LENGTH - 1); startOffset <= 0; startOffset += 1) {
                const start = {
                    x: anchor.x + direction[0] * startOffset,
                    y: anchor.y + direction[1] * startOffset,
                };
                const key = `${direction[0]},${direction[1]}:${start.x},${start.y}`;
                if (windows.has(key)) {
                    continue;
                }

                const window = buildLineWindow(occupiedByCell, playerId, start, direction);
                if (window && window.playerCount > 0) {
                    windows.set(key, window);
                }
            }
        }
    }

    return [...windows.values()];
}

function buildLineWindow(
    occupiedByCell: Map<string, string>,
    playerId: string,
    start: HexCoordinate,
    direction: Direction,
): LineWindow | null {
    const emptyCells: HexCoordinate[] = [];
    let playerCount = 0;

    for (let step = 0; step < WINNING_LINE_LENGTH; step += 1) {
        const cell = {
            x: start.x + direction[0] * step,
            y: start.y + direction[1] * step,
        };
        const occupant = occupiedByCell.get(getCellKey(cell.x, cell.y));
        if (!occupant) {
            emptyCells.push(cell);
            continue;
        }

        if (occupant !== playerId) {
            return null;
        }

        playerCount += 1;
    }

    return {
        emptyCells,
        playerCount,
    };
}

function applyTurnAction(gameState: GameState, action: TurnAction): GameState | null {
    const playerId = gameState.currentTurnPlayerId;
    if (!playerId) {
        return null;
    }

    const nextState = cloneGameState(gameState);
    for (const move of action.moves) {
        if (nextState.winner || nextState.currentTurnPlayerId !== playerId) {
            break;
        }

        try {
            applyGameMove(nextState, {
                playerId,
                x: move.x,
                y: move.y,
            });
        } catch {
            return null;
        }
    }

    return nextState;
}

function applyMoveToClone(gameState: GameState, playerId: string, move: HexCoordinate): GameState | null {
    const nextState = cloneGameState(gameState);
    try {
        applyGameMove(nextState, {
            playerId,
            x: move.x,
            y: move.y,
        });
    } catch {
        return null;
    }

    return nextState;
}

function isLegalMove(gameState: GameState, move: HexCoordinate): boolean {
    if (gameState.cells.some((cell) => cell.x === move.x && cell.y === move.y)) {
        return false;
    }

    if (gameState.cells.length === 0) {
        return move.x === 0 && move.y === 0;
    }

    return isCellWithinPlacementRadius(gameState.cells, move);
}

function getOpponentId(gameState: GameState, playerId: string | null): string | null {
    if (!playerId) {
        return null;
    }

    const playerIds = Object.keys(gameState.playerTiles);
    return playerIds.find((candidate) => candidate !== playerId)
        ?? gameState.cells.find((cell) => cell.occupiedBy !== playerId)?.occupiedBy
        ?? null;
}

function buildOccupiedMap(gameState: GameState): Map<string, string> {
    return new Map(gameState.cells.map((cell) => [
        getCellKey(cell.x, cell.y),
        cell.occupiedBy,
    ]));
}

function countAdjacentCells(
    gameState: GameState,
    playerId: string | null,
    move: HexCoordinate,
    radius: number,
): number {
    if (!playerId) {
        return 0;
    }

    return gameState.cells.filter((cell) =>
        cell.occupiedBy === playerId
        && getHexDistance(cell, move) <= radius).length;
}

function sameCell(left: HexCoordinate, right: HexCoordinate): boolean {
    return left.x === right.x && left.y === right.y;
}

function assertTimeRemaining(context: SearchContext): void {
    if (Date.now() >= context.deadline) {
        throw new SearchTimeout();
    }
}

export default async function () {
    return new MyBotEngine();
}
