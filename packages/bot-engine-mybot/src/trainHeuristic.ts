import { writeFileSync } from "node:fs";

import createSealEngine from "@ih3t/bot-engine-seal";
import type { BotEngineInterface, GameState, HexCoordinate } from "@ih3t/shared";
import { applyGameMove, cloneGameState, getCellKey, getHexDistance } from "@ih3t/shared";

import {
    chooseTrainingTurn,
    createInitialTrainingState,
    createTrainingSamples,
    LearnedValueFunction,
    TrainingSample,
} from "./index";
import persistedTrainingArtifact from "./learnedWeights.json";

const MAX_TURNS_PER_GAME = 18;
const DEFAULT_TRAINING_BUDGET_MS = 10_000;
const CHECKPOINT_INTERVAL_MS = 60_000;
const LEARNING_RATE = 0.006;
const TRAINING_EPOCHS = 2;
const LEARNED_WEIGHTS_URL = new URL(`./learnedWeights.json`, import.meta.url);
const LEARNER_PLAYER_ID = `learner`;
const SEAL_PLAYER_ID = `seal`;

const valueFunction = new LearnedValueFunction();
let totalSamples = 0;
let completedGames = 0;
let lastCheckpointAt = Date.now();
const trainingBudgetMs = readTrainingBudgetMs();
const trainingOpponent = readTrainingOpponent();
const previousTrainingSamples = readPreviousCount(persistedTrainingArtifact.trainingSamples);
const previousCompletedGames = readPreviousCount(persistedTrainingArtifact.completedGames);
const deadlineAt = Date.now() + trainingBudgetMs;
const sealEngine = trainingOpponent === `seal` ? await createSealEngine() : null;

/*
 * Offline trainer for improving DEFAULT_LEARNED_WEIGHTS.
 *
 * The live bot already performs a tiny amount of lazy training from the current
 * position. This script does the same idea at a larger scale: generate legal
 * self-play positions, label them with the deeper alpha-beta teacher, train the
 * linear value function, then save the updated weights for the engine to load.
 */
console.log(`Training for ${Math.round(trainingBudgetMs / 1000)} seconds against ${trainingOpponent}...`);

for (let gameIndex = 0; Date.now() < deadlineAt; gameIndex++) {
    const samples = sealEngine
        ? await collectSealGameSamples(sealEngine)
        : collectTeacherSelfPlaySamples(gameIndex);
    samples.push(...collectTacticalSamples());

    if (samples.length > 0) {
        valueFunction.train(samples, LEARNING_RATE, TRAINING_EPOCHS);
        totalSamples += samples.length;
    }

    completedGames += 1;

    if (Date.now() - lastCheckpointAt >= CHECKPOINT_INTERVAL_MS) {
        saveLearnedWeights();
        lastCheckpointAt = Date.now();
        console.log(`Checkpoint: ${completedGames} games, ${totalSamples} samples.`);
    }
}

const nextWeights = saveLearnedWeights();
console.log(`Completed games: ${completedGames}`);
console.log(`Training samples: ${totalSamples}`);
console.log(`Saved learned weights to ${LEARNED_WEIGHTS_URL.pathname}`);
console.log(JSON.stringify(nextWeights));

function readTrainingBudgetMs() {
    const ms = Number.parseInt(process.env.MYBOT_TRAINING_MS ?? ``, 10);
    if (Number.isFinite(ms) && ms > 0) {
        return ms;
    }

    const minutes = Number.parseFloat(process.env.MYBOT_TRAINING_MINUTES ?? ``);
    if (Number.isFinite(minutes) && minutes > 0) {
        return Math.round(minutes * 60_000);
    }

    return DEFAULT_TRAINING_BUDGET_MS;
}

function readTrainingOpponent(): `teacher` | `seal` {
    return process.env.MYBOT_TRAINING_OPPONENT === `seal` ? `seal` : `teacher`;
}

function readPreviousCount(value: number | undefined): number {
    return typeof value === `number` && Number.isFinite(value) && value > 0 ? value : 0;
}

function saveLearnedWeights() {
    const nextWeights = valueFunction.exportWeights().map(weight => Number(weight.toFixed(4)));
    const trainingArtifact = {
        version: 1,
        trainedAt: new Date().toISOString(),
        trainingSamples: previousTrainingSamples + totalSamples,
        completedGames: previousCompletedGames + completedGames,
        lastRunTrainingSamples: totalSamples,
        lastRunCompletedGames: completedGames,
        trainingBudgetMs,
        trainingOpponent,
        description: `Persisted linear value-function weights for MyBot. Run \`corepack pnpm --filter @ih3t/bot-engine-mybot train:heuristic\` to regenerate.`,
        weights: nextWeights,
    };

    writeFileSync(LEARNED_WEIGHTS_URL, `${JSON.stringify(trainingArtifact, null, 4)}\n`);
    return nextWeights;
}

function collectTeacherSelfPlaySamples(gameIndex: number): TrainingSample[] {
    let gameState = createInitialTrainingState();
    const samples: TrainingSample[] = [];

    for (let turnIndex = 0; turnIndex < MAX_TURNS_PER_GAME && !gameState.winner && Date.now() < deadlineAt; turnIndex++) {
        const playerId = gameState.currentTurnPlayerId;
        if (!playerId) {
            break;
        }

        samples.push(...createTrainingSamples(gameState, playerId, deadlineAt));
        gameState = advanceDeterministicSelfPlay(gameState, gameIndex + turnIndex);
    }

    return samples;
}

function collectTacticalSamples(): TrainingSample[] {
    return [
        createLearnerFiveThreatState(),
        createSealFiveThreatState(),
    ].flatMap(gameState => createTrainingSamples(gameState, LEARNER_PLAYER_ID, deadlineAt));
}

async function collectSealGameSamples(sealBot: BotEngineInterface): Promise<TrainingSample[]> {
    let gameState = createSealTrainingState();
    const samples: TrainingSample[] = [];

    for (let turnIndex = 0; turnIndex < MAX_TURNS_PER_GAME && !gameState.winner && Date.now() < deadlineAt; turnIndex++) {
        samples.push(...createTrainingSamples(gameState, LEARNER_PLAYER_ID, deadlineAt));

        const playerId = gameState.currentTurnPlayerId;
        if (!playerId) {
            break;
        }

        const moves = playerId === LEARNER_PLAYER_ID
            ? chooseTrainingTurn(gameState, valueFunction, deadlineAt)
            : await getSealTurn(sealBot, gameState);
        gameState = applyTrainingMoves(gameState, moves);
    }

    return samples;
}

function createSealTrainingState(): GameState {
    const gameState = createInitialTrainingState();
    gameState.playerTiles = {
        [LEARNER_PLAYER_ID]: gameState.playerTiles.learner,
        [SEAL_PLAYER_ID]: gameState.playerTiles.opponent,
    };
    gameState.currentTurnPlayerId = SEAL_PLAYER_ID;

    return gameState;
}

function createLearnerFiveThreatState(): GameState {
    const gameState = createSealTrainingState();

    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 0, y: 1 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 0, y: 2 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 1, y: 0 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 2, y: 0 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 0, y: 3 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 0, y: 4 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 3, y: 0 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 4, y: 0 });

    return gameState;
}

function createSealFiveThreatState(): GameState {
    const gameState = createSealTrainingState();

    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 1, y: 0 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 2, y: 0 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 0, y: 1 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 0, y: 2 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 3, y: 0 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 4, y: 0 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 0, y: 3 });
    applyGameMove(gameState, { playerId: LEARNER_PLAYER_ID, x: 0, y: 4 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 5, y: 0 });
    applyGameMove(gameState, { playerId: SEAL_PLAYER_ID, x: 2, y: 1 });

    return gameState;
}

async function getSealTurn(sealBot: BotEngineInterface, gameState: GameState): Promise<HexCoordinate[]> {
    const originalConsoleLog = console.log;
    console.log = () => undefined;

    try {
        const result = await sealBot.suggestTurn(cloneGameState(gameState), 1_500);
        return result.status === `provide`
            ? result.suggestion
            : createFallbackTrainingTurn(gameState);
    } catch {
        return createFallbackTrainingTurn(gameState);
    } finally {
        console.log = originalConsoleLog;
    }
}

function applyTrainingMoves(gameState: GameState, moves: readonly HexCoordinate[]): GameState {
    const nextState = cloneGameState(gameState);
    const fallbackMoves = moves.length > 0 ? moves : createFallbackTrainingTurn(nextState);

    for (const move of fallbackMoves) {
        const playerId = nextState.currentTurnPlayerId;
        if (!playerId || nextState.winner) {
            break;
        }

        applyGameMove(nextState, { playerId, x: move.x, y: move.y });
    }

    return nextState;
}

function createFallbackTrainingTurn(gameState: GameState): HexCoordinate[] {
    return createLocalCandidateMoves(gameState).slice(0, gameState.placementsRemaining);
}

function advanceDeterministicSelfPlay(
    gameState: GameState,
    offset: number,
): GameState {
    const nextState = cloneGameState(gameState);
    const playerId = nextState.currentTurnPlayerId;
    if (!playerId) {
        return nextState;
    }

    const legalMoves = createLocalCandidateMoves(nextState);
    const moveCount = nextState.placementsRemaining;

    for (const move of legalMoves.slice(offset % 3, (offset % 3) + moveCount)) {
        if (nextState.winner || nextState.currentTurnPlayerId !== playerId) {
            break;
        }

        applyGameMove(nextState, { playerId, x: move.x, y: move.y });
    }

    return nextState;
}

function createLocalCandidateMoves(gameState: GameState) {
    const occupied = new Set(gameState.cells.map(cell => getCellKey(cell.x, cell.y)));
    const moves = new Map<string, { x: number, y: number }>();

    for (const anchor of gameState.cells) {
        for (let dx = -2; dx <= 2; dx++) {
            for (let dy = -2; dy <= 2; dy++) {
                const move = { x: anchor.x + dx, y: anchor.y + dy };
                const key = getCellKey(move.x, move.y);
                if (!occupied.has(key) && getHexDistance(anchor, move) <= 2) {
                    moves.set(key, move);
                }
            }
        }
    }

    return [...moves.values()].slice(0, 12);
}
