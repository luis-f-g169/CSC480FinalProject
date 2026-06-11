import {
    applyGameMove,
    BotEngineCapabilities,
    BotEngineInterface,
    BotEngineSuggestionResult,
    cloneGameState,
    createStartedGameState,
    GameState,
    getCellKey,
    getHexDistance,
    HexCoordinate,
    isCellWithinPlacementRadius,
} from "@ih3t/shared";

import learnedWeights from "./learnedWeights.json";

export { default as createMdpBellmanBotEngine } from "./mdpBellmanBot";

const WINNING_LINE_LENGTH = 6;
const DIRECTIONS: readonly HexCoordinate[] = [
    { x: 1, y: 0 },
    { x: 0, y: 1 },
    { x: 1, y: -1 },
];

const SEARCH_NEIGHBOR_RADIUS = 3;
const LINE_PROJECTION_RADIUS = 5;
const FIRST_MOVE_LIMIT = 16;
const SECOND_MOVE_LIMIT = 12;
const MAX_TURNS_PER_NODE = 80;
const TRAINING_SAMPLE_LIMIT = 18;
const TRAINING_ROLLOUT_LIMIT = 3;
const TEACHER_SEARCH_DEPTH = 2;

type ScoredMove = {
    move: HexCoordinate;
    score: number;
};

type ScoredTurn = {
    moves: HexCoordinate[];
    score: number;
};

type SearchContext = {
    perspectivePlayerId: string;
    deadlineAt: number;
    nodes: number;
    timedOut: boolean;
};

export type TrainingSample = {
    features: number[];
    target: number;
};

type FeatureStats = {
    longest: number;
    openThree: number;
    openFour: number;
    openFive: number;
    closedFive: number;
    centerWeight: number;
};

type LineDetails = {
    length: number;
    openEnds: number;
};

const DEFAULT_LEARNED_WEIGHTS = [
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

export class LearnedValueFunction {
    private readonly weights: number[];

    constructor(initialWeights = getInitialLearnedWeights()) {
        this.weights = [...initialWeights];
    }

    evaluate(gameState: GameState, playerId: string): number {
        const terminalValue = getTerminalValue(gameState, playerId);
        if (terminalValue !== null) {
            return terminalValue;
        }

        const learnedScore = squash(dot(this.weights, extractFeatures(gameState, playerId)));
        const tacticalScore = staticTacticalEvaluation(gameState, playerId);

        return clampEvaluation((learnedScore * 0.65) + (tacticalScore * 0.35));
    }

    train(samples: readonly TrainingSample[], learningRate = 0.04, epochs = 4): void {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (const sample of samples) {
                if (!Number.isFinite(sample.target) || sample.features.some(feature => !Number.isFinite(feature))) {
                    continue;
                }

                const prediction = squash(dot(this.weights, sample.features));
                if (!Number.isFinite(prediction)) {
                    continue;
                }

                const gradientScale = (sample.target - prediction) * (1 - prediction * prediction);

                for (let index = 0; index < this.weights.length; index++) {
                    this.weights[index] += learningRate * gradientScale * (sample.features[index] ?? 0);
                }
            }
        }
    }

    exportWeights(): readonly number[] {
        return [...this.weights];
    }
}

export function chooseTrainingTurn(
    gameState: GameState,
    valueFunction: LearnedValueFunction,
    deadlineAt: number,
): HexCoordinate[] {
    const playerId = gameState.currentTurnPlayerId;
    if (!playerId) {
        return [];
    }

    const context = createSearchContext(playerId, deadlineAt);
    return chooseBestTurn(
        gameState,
        context,
        TEACHER_SEARCH_DEPTH,
        (state, perspectivePlayerId) => valueFunction.evaluate(state, perspectivePlayerId),
    )?.moves ?? [];
}

function getInitialLearnedWeights(): readonly number[] {
    const persistedWeights = learnedWeights.weights;
    if (
        Array.isArray(persistedWeights)
        && persistedWeights.length === DEFAULT_LEARNED_WEIGHTS.length
        && persistedWeights.every(weight => Number.isFinite(weight))
    ) {
        return persistedWeights;
    }

    return DEFAULT_LEARNED_WEIGHTS;
}

class MyBotEngine implements BotEngineInterface {
    private readonly valueFunction = new LearnedValueFunction();
    private trainingRuns = 0;

    getDisplayName(): string {
        return `MyBot Learning Alpha-Beta`;
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
        const botPlayerId = gameState.currentTurnPlayerId;
        if (!botPlayerId) {
            return { status: `failure`, message: `No current player to move.`, metadata: {} };
        }

        const deadlineAt = Date.now() + Math.max(50, timeoutMs - 20);
        if (timeoutMs >= 3_000) {
            this.trainFromCurrentPosition(gameState, botPlayerId, deadlineAt);
        }

        const context = createSearchContext(botPlayerId, deadlineAt);
        const bestTurn = chooseBestTurn(
            gameState,
            context,
            getSearchDepth(timeoutMs),
            (state, playerId) => this.valueFunction.evaluate(state, playerId),
        );

        const suggestion = padTurnSuggestion(gameState, botPlayerId, bestTurn?.moves ?? []);
        return {
            status: `provide`,
            suggestion,
            metadata: {
                nodes: String(context.nodes),
                searchTimedOut: String(context.timedOut),
                trainedPositions: String(this.trainingRuns),
                weights: this.valueFunction.exportWeights()
                    .map(weight => weight.toFixed(3))
                    .join(`,`),
            },
        };
    }

    async suggestMove(
        _gameState: GameState,
        _timeoutMs: number,
    ): Promise<BotEngineSuggestionResult<HexCoordinate>> {
        return { status: `failure`, message: `MyBot searches complete two-placement turns only.`, metadata: {} };
    }

    shutdown(): void {
        /* The learned weights live in memory and do not need cleanup. */
    }

    private trainFromCurrentPosition(gameState: GameState, playerId: string, deadlineAt: number): void {
        if (this.trainingRuns >= 4 || Date.now() >= deadlineAt) {
            return;
        }

        const samples = createTrainingSamples(gameState, playerId, deadlineAt);
        if (samples.length === 0) {
            return;
        }

        this.valueFunction.train(samples);
        this.trainingRuns += samples.length;
    }
}

function chooseBestTurn(
    gameState: GameState,
    context: SearchContext,
    depth: number,
    evaluator: (state: GameState, playerId: string) => number,
): ScoredTurn | null {
    const playerId = gameState.currentTurnPlayerId;
    if (!playerId) {
        return null;
    }

    let bestTurn: ScoredTurn | null = null;
    let alpha = Number.NEGATIVE_INFINITY;

    for (const turn of generateCandidateTurns(gameState, playerId)) {
        if (hasSearchTimedOut(context)) {
            break;
        }

        const nextState = applyTurn(gameState, playerId, turn.moves);
        const value = alphaBetaValue(nextState, depth - 1, alpha, Number.POSITIVE_INFINITY, context, evaluator);
        const score = value + turn.score * 0.001;

        if (!bestTurn || score > bestTurn.score) {
            bestTurn = { moves: turn.moves, score };
        }

        alpha = Math.max(alpha, score);
    }

    return bestTurn;
}

function alphaBetaValue(
    gameState: GameState,
    depth: number,
    alpha: number,
    beta: number,
    context: SearchContext,
    evaluator: (state: GameState, playerId: string) => number,
): number {
    context.nodes += 1;

    const terminalValue = getTerminalValue(gameState, context.perspectivePlayerId);
    if (terminalValue !== null) {
        return terminalValue;
    }

    if (depth <= 0 || hasSearchTimedOut(context) || !gameState.currentTurnPlayerId) {
        return evaluator(gameState, context.perspectivePlayerId);
    }

    const currentPlayerId = gameState.currentTurnPlayerId;
    const maximizing = currentPlayerId === context.perspectivePlayerId;
    const candidateTurns = generateCandidateTurns(gameState, currentPlayerId);
    if (candidateTurns.length === 0) {
        return evaluator(gameState, context.perspectivePlayerId);
    }

    let bestValue = maximizing ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;

    for (const turn of candidateTurns) {
        if (hasSearchTimedOut(context)) {
            break;
        }

        const nextState = applyTurn(gameState, currentPlayerId, turn.moves);
        const value = alphaBetaValue(nextState, depth - 1, alpha, beta, context, evaluator);

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

    return Number.isFinite(bestValue)
        ? bestValue
        : evaluator(gameState, context.perspectivePlayerId);
}

function generateCandidateTurns(gameState: GameState, playerId: string): ScoredTurn[] {
    const firstMoves = scoreLegalMoves(gameState, playerId, FIRST_MOVE_LIMIT);
    const turns: ScoredTurn[] = [];

    for (const firstMove of firstMoves) {
        const afterFirstMove = applyTurn(gameState, playerId, [firstMove.move]);
        if (afterFirstMove.winner || afterFirstMove.currentTurnPlayerId !== playerId) {
            turns.push({ moves: [firstMove.move], score: firstMove.score + 10_000 });
            continue;
        }

        const secondMoves = scoreLegalMoves(afterFirstMove, playerId, SECOND_MOVE_LIMIT);
        for (const secondMove of secondMoves) {
            turns.push({
                moves: [firstMove.move, secondMove.move],
                score: firstMove.score + secondMove.score,
            });
        }
    }

    return dedupeTurns(turns)
        .sort((a, b) => b.score - a.score)
        .slice(0, MAX_TURNS_PER_NODE);
}

function scoreLegalMoves(gameState: GameState, playerId: string, limit: number): ScoredMove[] {
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

    const candidateMap = new Map<string, HexCoordinate>();

    // The real board is infinite, so search only the tactically relevant frontier.
    for (const anchor of gameState.cells) {
        for (let dx = -SEARCH_NEIGHBOR_RADIUS; dx <= SEARCH_NEIGHBOR_RADIUS; dx++) {
            for (let dy = -SEARCH_NEIGHBOR_RADIUS; dy <= SEARCH_NEIGHBOR_RADIUS; dy++) {
                const candidate = { x: anchor.x + dx, y: anchor.y + dy };
                const key = getCellKey(candidate.x, candidate.y);
                if (
                    occupied.has(key)
                    || getHexDistance(anchor, candidate) > SEARCH_NEIGHBOR_RADIUS
                    || !isCellWithinPlacementRadius(gameState.cells, candidate)
                ) {
                    continue;
                }

                candidateMap.set(key, candidate);
            }
        }
    }

    /*
     * Also project along the three winning axes. This catches urgent endpoints
     * and one-cell gaps that can sit outside the local radius around a stone.
     */
    for (const anchor of gameState.cells) {
        for (const direction of DIRECTIONS) {
            for (let step = -LINE_PROJECTION_RADIUS; step <= LINE_PROJECTION_RADIUS; step++) {
                if (step === 0) {
                    continue;
                }

                const candidate = {
                    x: anchor.x + direction.x * step,
                    y: anchor.y + direction.y * step,
                };
                const key = getCellKey(candidate.x, candidate.y);
                if (
                    occupied.has(key)
                    || !isCellWithinPlacementRadius(gameState.cells, candidate)
                ) {
                    continue;
                }

                candidateMap.set(key, candidate);
            }
        }
    }

    if (candidateMap.size > 0) {
        return [...candidateMap.values()];
    }

    // Fallback for unusual imported positions where every near-frontier point is occupied.
    for (const anchor of gameState.cells) {
        for (let dx = -8; dx <= 8; dx++) {
            for (let dy = -8; dy <= 8; dy++) {
                const candidate = { x: anchor.x + dx, y: anchor.y + dy };
                const key = getCellKey(candidate.x, candidate.y);
                if (
                    !occupied.has(key)
                    && getHexDistance(anchor, candidate) <= 8
                    && isCellWithinPlacementRadius(gameState.cells, candidate)
                ) {
                    candidateMap.set(key, candidate);
                }
            }
        }
    }

    return [...candidateMap.values()];
}

function scoreMove(gameState: GameState, playerId: string, move: HexCoordinate): number {
    const opponentId = getOpponentPlayerId(gameState, playerId);
    const ownCells = getPlayerCellSet(gameState, playerId);
    const opponentCells = opponentId ? getPlayerCellSet(gameState, opponentId) : new Set<string>();
    const occupied = new Set(gameState.cells.map(cell => getCellKey(cell.x, cell.y)));

    const ownLine = getBestLineDetailsThroughMove(ownCells, occupied, move);
    const opponentLine = getBestLineDetailsThroughMove(opponentCells, occupied, move);

    if (ownLine.length >= WINNING_LINE_LENGTH) {
        return 1_000_000;
    }

    if (opponentLine.length >= WINNING_LINE_LENGTH) {
        return 900_000;
    }

    // Prefer wins, urgent blocks, and moves that keep stones near live formations.
    return scoreLineDetails(ownLine, 1)
        + scoreLineDetails(opponentLine, 1.15)
        - Math.abs(move.x) * 0.04
        - Math.abs(move.y) * 0.04
        - Math.abs(move.x + move.y) * 0.02;
}

function applyTurn(gameState: GameState, playerId: string, moves: readonly HexCoordinate[]): GameState {
    const nextState = cloneGameState(gameState);

    for (const move of moves) {
        if (nextState.winner || nextState.currentTurnPlayerId !== playerId) {
            break;
        }

        applyGameMove(nextState, {
            playerId,
            x: move.x,
            y: move.y,
        });
    }

    return nextState;
}

function padTurnSuggestion(
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

        applyGameMove(nextState, {
            playerId,
            x: move.x,
            y: move.y,
        });
    }

    while (paddedMoves.length < 2) {
        if (nextState.winner) {
            paddedMoves.push(paddedMoves[0] ?? { x: 0, y: 0 });
            continue;
        }

        const fallback = scoreLegalMoves(nextState, playerId, 1)[0]?.move ?? { x: 0, y: 0 };
        paddedMoves.push(fallback);
        nextState = applyTurn(nextState, playerId, [fallback]);
    }

    return [paddedMoves[0], paddedMoves[1]];
}

export function createTrainingSamples(gameState: GameState, playerId: string, deadlineAt: number): TrainingSample[] {
    const samples: TrainingSample[] = [];
    const seed = hashPosition(gameState) + playerId.length;
    const random = createSeededRandom(seed);
    let rolloutState = cloneGameState(gameState);

    for (let sampleIndex = 0; sampleIndex < TRAINING_SAMPLE_LIMIT && Date.now() < deadlineAt; sampleIndex++) {
        const target = labelWithTeacherSearch(rolloutState, playerId, deadlineAt);
        const features = extractFeatures(rolloutState, playerId);
        if (Number.isFinite(target) && features.every(feature => Number.isFinite(feature))) {
            samples.push({ features, target });
        }

        rolloutState = advanceTrainingRollout(rolloutState, random);
        if (rolloutState.winner || sampleIndex % TRAINING_ROLLOUT_LIMIT === TRAINING_ROLLOUT_LIMIT - 1) {
            rolloutState = cloneGameState(gameState);
        }
    }

    return samples;
}

function labelWithTeacherSearch(gameState: GameState, playerId: string, deadlineAt: number): number {
    const terminalValue = getTerminalValue(gameState, playerId);
    if (terminalValue !== null) {
        return terminalValue;
    }

    const context = createSearchContext(playerId, deadlineAt);
    return alphaBetaValue(
        gameState,
        TEACHER_SEARCH_DEPTH,
        Number.NEGATIVE_INFINITY,
        Number.POSITIVE_INFINITY,
        context,
        staticTacticalEvaluation,
    );
}

function advanceTrainingRollout(gameState: GameState, random: () => number): GameState {
    const playerId = gameState.currentTurnPlayerId;
    if (!playerId || gameState.winner) {
        return gameState;
    }

    const turns = generateCandidateTurns(gameState, playerId);
    const chosenTurn = turns[Math.min(turns.length - 1, Math.floor(random() * Math.min(5, turns.length)))] ?? null;
    return chosenTurn ? applyTurn(gameState, playerId, chosenTurn.moves) : gameState;
}

function extractFeatures(gameState: GameState, playerId: string): number[] {
    const opponentId = getOpponentPlayerId(gameState, playerId);
    const ownStats = collectFeatureStats(gameState, playerId);
    const opponentStats = opponentId
        ? collectFeatureStats(gameState, opponentId)
        : createEmptyFeatureStats();

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

function staticTacticalEvaluation(gameState: GameState, playerId: string): number {
    const opponentId = getOpponentPlayerId(gameState, playerId);
    const ownStats = collectFeatureStats(gameState, playerId);
    const opponentStats = opponentId
        ? collectFeatureStats(gameState, opponentId)
        : createEmptyFeatureStats();

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

    return squash(score);
}

function collectFeatureStats(gameState: GameState, playerId: string): FeatureStats {
    const playerCells = gameState.cells.filter(cell => cell.occupiedBy === playerId);
    const playerSet = new Set(playerCells.map(cell => getCellKey(cell.x, cell.y)));
    const occupied = new Set(gameState.cells.map(cell => getCellKey(cell.x, cell.y)));
    const stats = createEmptyFeatureStats();

    for (const cell of playerCells) {
        stats.centerWeight += 1 / (1 + getHexDistance(cell, { x: 0, y: 0 }));

        for (const direction of DIRECTIONS) {
            const previousKey = getCellKey(cell.x - direction.x, cell.y - direction.y);
            if (playerSet.has(previousKey)) {
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

        if (
            length > bestDetails.length
            || (length === bestDetails.length && openEnds > bestDetails.openEnds)
        ) {
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

function dedupeTurns(turns: readonly ScoredTurn[]): ScoredTurn[] {
    const seen = new Set<string>();
    const dedupedTurns: ScoredTurn[] = [];

    for (const turn of turns) {
        const key = turn.moves
            .map(move => getCellKey(move.x, move.y))
            .sort()
            .join(`|`);

        if (seen.has(key)) {
            continue;
        }

        seen.add(key);
        dedupedTurns.push(turn);
    }

    return dedupedTurns;
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

function createSearchContext(perspectivePlayerId: string, deadlineAt: number): SearchContext {
    return {
        perspectivePlayerId,
        deadlineAt,
        nodes: 0,
        timedOut: false,
    };
}

function hasSearchTimedOut(context: SearchContext): boolean {
    if (Date.now() < context.deadlineAt) {
        return false;
    }

    context.timedOut = true;
    return true;
}

function getSearchDepth(timeoutMs: number): number {
    if (timeoutMs >= 12_000) {
        return 4;
    }

    if (timeoutMs >= 3_000) {
        return 3;
    }

    return 2;
}

function dot(weights: readonly number[], features: readonly number[]): number {
    return weights.reduce((total, weight, index) => total + weight * (features[index] ?? 0), 0);
}

function squash(value: number): number {
    return Math.tanh(value);
}

function clampEvaluation(value: number): number {
    return Math.max(-1, Math.min(1, value));
}

function hashPosition(gameState: GameState): number {
    return gameState.cells.reduce(
        (hash, cell) => Math.imul(hash ^ (cell.x * 31 + cell.y * 17), 16_777_619),
        2_166_136_261,
    );
}

function createSeededRandom(seed: number): () => number {
    let state = seed >>> 0;

    return () => {
        state += 0x6D2B79F5;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
    };
}

export function createInitialTrainingState(): GameState {
    const gameState = createStartedGameState([`learner`, `opponent`], `learner`);
    applyGameMove(gameState, {
        playerId: `learner`,
        x: 0,
        y: 0,
    });

    return gameState;
}

export default async function () {
    return new MyBotEngine();
}
