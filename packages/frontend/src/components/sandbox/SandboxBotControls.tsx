import type { BotEngineCapabilities, SandboxPlayerSlot } from '@ih3t/shared';
import React, { useEffect, useState } from 'react';

import type { SandboxPlayerMode } from '../../sandbox/sandboxBotSettings';

type SandboxBotControlsProps = {
    botCapabilities: Readonly<BotEngineCapabilities> | null
    botAvailabilityMessage: string | null
    selectedEngineLabels: Record<SandboxPlayerSlot, string | null>
    playerModes: Record<SandboxPlayerSlot, SandboxPlayerMode>
    currentTurnPlayerSlot: SandboxPlayerSlot | null
    timeoutMs: number
    isBotThinking: boolean
    isCurrentTurnBotControlled: boolean
    botErrorMessage: string | null
    onPlayerModeChange: (playerSlot: SandboxPlayerSlot, nextMode: SandboxPlayerMode) => void
    onChangePlayerEngine: (playerSlot: SandboxPlayerSlot) => void
    onStartBotMatch: () => void
    onTimeoutMsChange: (timeoutMs: number) => void
};

const PLAYER_OPTIONS: readonly {
    slot: SandboxPlayerSlot
    title: string
    subtitle: string
}[] = [
    {
        slot: `player-1`,
        title: `Player 1`,
        subtitle: `Opens the game at the origin.`,
    },
    {
        slot: `player-2`,
        title: `Player 2`,
        subtitle: `Responds after the first turn.`,
    },
];

function SandboxBotControls({
    botAvailabilityMessage,
    selectedEngineLabels,
    playerModes,
    currentTurnPlayerSlot,
    timeoutMs,
    botErrorMessage,
    onPlayerModeChange,
    onChangePlayerEngine,
    onStartBotMatch,
    onTimeoutMsChange,
}: Readonly<SandboxBotControlsProps>) {
    const hasPlayerOneEngine = Boolean(selectedEngineLabels[`player-1`]);
    const hasPlayerTwoEngine = Boolean(selectedEngineLabels[`player-2`]);
    const canStartBotMatch = hasPlayerOneEngine && hasPlayerTwoEngine;

    const [timeoutMsText, setTimeoutMsText] = useState<string | null>(null);

    useEffect(() => {
        if (!timeoutMsText) {
            return;
        }

        const id = setTimeout(
            () => {
                setTimeoutMsText(null);
                onTimeoutMsChange(Number.parseInt(timeoutMsText, 10));
            },
            1000,
        );
        return () => clearTimeout(id);
    }, [timeoutMsText]);

    return (
        <React.Fragment>

            {botAvailabilityMessage && (
                <div className="mt-3 rounded-[0.9rem] border border-amber-300/20 bg-amber-300/10 px-3 py-2.5 text-xs leading-5 text-amber-50/85">
                    {botAvailabilityMessage}
                </div>
            )}

            {botErrorMessage && (
                <div className="mt-3 rounded-[0.9rem] border border-rose-300/20 bg-rose-300/10 px-3 py-2.5 text-xs leading-5 text-rose-50/90">
                    {botErrorMessage}
                </div>
            )}

            <button
                type="button"
                disabled={!canStartBotMatch}
                onClick={onStartBotMatch}
                className={`mt-3 w-full rounded-[0.9rem] border px-3 py-2.5 text-sm font-semibold transition ${canStartBotMatch
                    ? `border-sky-300/35 bg-sky-300/12 text-white hover:bg-sky-300/18`
                    : `cursor-not-allowed border-white/8 bg-white/4 text-slate-500`
                }`}
            >
                Start Bot vs Bot
            </button>

            {!canStartBotMatch && (
                <div className="mt-2 text-[11px] leading-5 text-slate-300">
                    Choose an engine for both players before starting a bot match.
                </div>
            )}

            <div className="mt-3 grid gap-2 transition">
                {PLAYER_OPTIONS.map((playerOption) => {
                    const isCurrentTurn = currentTurnPlayerSlot === playerOption.slot;
                    const selectedMode = playerModes[playerOption.slot];
                    const selectedEngineLabel = selectedEngineLabels[playerOption.slot];
                    const botButtonDisabled = !selectedEngineLabel;

                    return (
                        <div
                            key={playerOption.slot}
                            className={`rounded-[0.9rem] border px-3 py-3 ${isCurrentTurn
                                ? `border-sky-300/25 bg-sky-300/8`
                                : `border-white/10 bg-white/5`
                            }`}
                        >
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <div className="text-sm font-semibold text-white">
                                        {playerOption.title}
                                    </div>

                                    <div className="text-[11px] leading-4.5 text-slate-300">
                                        {playerOption.subtitle}
                                    </div>
                                </div>

                                {isCurrentTurn && (
                                    <div className="rounded-full border border-white/10 bg-white/8 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-100">
                                        To Move
                                    </div>
                                )}
                            </div>

                            <div className="mt-3 flex items-center justify-between gap-3 rounded-[0.8rem] border border-white/10 bg-white/5 px-3 py-2">
                                <div>
                                    <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">
                                        Engine
                                    </div>

                                    <div className="mt-0.5 text-xs font-medium text-slate-100">
                                        {selectedEngineLabel ?? `None selected`}
                                    </div>
                                </div>

                                <button
                                    type="button"
                                    onClick={() => onChangePlayerEngine(playerOption.slot)}
                                    className="rounded-full border border-white/12 bg-white/8 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-white/12"
                                >
                                    Change
                                </button>
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-2">
                                <button
                                    type="button"
                                    onClick={() => onPlayerModeChange(playerOption.slot, `human`)}
                                    className={`rounded-[0.9rem] border px-3 py-2 text-sm font-medium transition ${selectedMode === `human`
                                        ? `border-emerald-300/35 bg-emerald-300/10 text-white`
                                        : `border-white/10 bg-white/6 text-slate-200 hover:bg-white/10`
                                    }`}
                                >
                                    Human
                                </button>

                                <button
                                    type="button"
                                    onClick={() => onPlayerModeChange(playerOption.slot, `bot`)}
                                    disabled={botButtonDisabled}
                                    className={`rounded-[0.9rem] border px-3 py-2 text-sm font-medium transition ${selectedMode === `bot`
                                        ? `border-sky-300/35 bg-sky-300/10 text-white`
                                        : botButtonDisabled
                                            ? `cursor-not-allowed border-white/8 bg-white/4 text-slate-500`
                                            : `border-white/10 bg-white/6 text-slate-200 hover:bg-white/10`
                                    }`}
                                >
                                    Bot
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="mt-3 rounded-[0.9rem] border border-white/10 bg-white/5 px-3 py-3 transition">
                <label className="block text-[11px] uppercase tracking-[0.22em] text-slate-400" htmlFor="sandbox-bot-timeout">
                    Timeout Per Request
                </label>

                <div className="mt-2 flex items-center gap-2">
                    <input
                        id="sandbox-bot-timeout"
                        type="number"

                        min={100}
                        max={60000}
                        step={100}
                        value={timeoutMsText ?? timeoutMs}

                        onChange={(event) => setTimeoutMsText(event.target.value)}
                        onBlur={() => {
                            if (!timeoutMsText) {
                                return;
                            }

                            setTimeoutMsText(null);
                            onTimeoutMsChange(Number.parseInt(timeoutMsText, 10));
                        }}

                        className="w-full rounded-[0.8rem] border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-white outline-none transition focus:border-sky-300/35"
                    />

                    <div className="rounded-[0.8rem] border border-white/10 bg-white/6 px-3 py-2 text-sm text-slate-200">
                        ms
                    </div>
                </div>

                <div className="mt-2 text-[11px] leading-5 text-slate-300">
                    Single-move bots may use this budget more than once during the same turn.
                </div>
            </div>
        </React.Fragment>
    );
}

export default SandboxBotControls;
