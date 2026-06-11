import {
    BotEngineCapabilities,
    BotEngineInterface,
    BotEngineSuggestionResult,
    GameState,
    getCellKey,
    HexCoordinate,
    isCellWithinPlacementRadius,
} from "@ih3t/shared";

class MyBotEngine implements BotEngineInterface {
    getDisplayName(): string {
        return "MyBot";
    }

    getCapabilities(): Readonly<BotEngineCapabilities> {
        return {
            suggestTurn: true,
            suggestMove: false,
        };
    }

    async suggestTurn(
        gameState: GameState,
        _timeoutMs: number
    ): Promise<BotEngineSuggestionResult<[HexCoordinate, HexCoordinate]>> {
        const botPlayerId = gameState.currentTurnPlayerId!;
        const occupied = new Set(gameState.cells.map(c => getCellKey(c.x, c.y)));

        // Helper: find all valid candidate cells
        const getCandidates = (): HexCoordinate[] => {
            const candidates: HexCoordinate[] = [];
            const radius = 10; // search window around origin
            for (let x = -radius; x <= radius; x++) {
                for (let y = -radius; y <= radius; y++) {
                    if (
                        !occupied.has(getCellKey(x, y)) &&
                        isCellWithinPlacementRadius(gameState.cells, { x, y })
                    ) {
                        candidates.push({ x, y });
                    }
                }
            }
            return candidates;
        };

        // --- YOUR STRATEGY GOES HERE ---
        // Ideas to implement:
        //   - Score each candidate by how many of your own cells are nearby
        //   - Block opponent's longest line if it's close to 6
        //   - Prefer cells that extend your own longest line
        //   - Minimax with alpha-beta pruning for deeper lookahead
        const candidates = getCandidates();
        const move1 = candidates[0] ?? { x: 0, y: 0 };

        // After picking move1, mark it occupied so move2 doesn't collide
        occupied.add(getCellKey(move1.x, move1.y));
        const move2 = getCandidates()[0] ?? { x: 1, y: 0 };

        return {
            status: "provide",
            suggestion: [move1, move2],
            metadata: {},
        };
    }

    async suggestMove(
        _gameState: GameState,
        _timeoutMs: number
    ): Promise<BotEngineSuggestionResult<HexCoordinate>> {
        return { status: "failure", message: "not supported", metadata: {} };
    }

    shutdown(): void {}
}

export default async function () {
    return new MyBotEngine();
}
