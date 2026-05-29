import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone

from livekit import api as lkapi
from livekit import rtc
from livekit.agents import AgentSession, JobContext, function_tool

logger = logging.getLogger("agent")


class CallState:
    def __init__(self):
        self.hangup_initiated = False
        self.survey_responses: dict[str, str] = {}
        self.events: list[dict] = []
        # Answer extraction state — managed by on_item_added in agent.py
        self.q_idx: int = 0                          # index of next question to be answered
        self.pending_question_key: str | None = None  # key of the question currently asked by agent


def _add_event(state: CallState, event_type: str, **fields: object) -> None:
    """Append a timestamped event. Never raises — must not break call flow."""
    with suppress(Exception):
        state.events.append({
            "type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
            **fields,
        })


async def _disconnect_participant(room_name: str, participant_identity: str):
    try:
        lk = lkapi.LiveKitAPI()
        await lk.room.remove_participant(
            lkapi.RoomParticipantIdentity(room=room_name, identity=participant_identity)
        )
        await lk.aclose()
        logger.info("[hangup] participant removed")
    except Exception as e:
        logger.warning(f"[hangup] remove failed: {e}")


async def perform_hangup(
    ctx: JobContext,
    session: AgentSession,
    participant: rtc.RemoteParticipant,
    state: CallState,
):
    if state.hangup_initiated:
        return
    state.hangup_initiated = True
    _add_event(state, "hangup_initiated")
    logger.info("[hangup] starting")
    await asyncio.sleep(1.2)
    await _disconnect_participant(ctx.room.name, participant.identity)
    with suppress(Exception):
        await session.aclose()
    with suppress(Exception):
        await ctx.room.disconnect()
    logger.info("[hangup] done")


def build_tools(
    ctx: JobContext,
    session: AgentSession,
    participant: rtc.RemoteParticipant,
    state: CallState,
):
    @function_tool(
        name="end_call",
        description=(
            "End the phone call. Call ONLY after saying a warm farewell and all "
            "survey questions are done. Never speak after calling this."
        ),
    )
    async def end_call() -> str:
        if state.hangup_initiated:
            return ""
        logger.info("[end_call] triggered")
        hangup_task = asyncio.ensure_future(
            perform_hangup(ctx, session, participant, state)
        )
        hangup_task.add_done_callback(lambda _: None)
        return ""

    return [end_call]
