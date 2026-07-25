"""Chat — transcript fetching, participant analysis, and the chat entry point."""

import atlantis
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .game import require_membership


logger = logging.getLogger("mcp_client")


def analyze_participants(raw_transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize transcript participants"""
    participants: Dict[str, Any] = {}
    last_speaker: Optional[str] = None

    for msg in raw_transcript:
        if msg.get('type') != 'chat':
            continue
        sid = msg.get('sid')
        if not sid or sid == 'system':
            continue
        if 'thinking' in str(msg.get('who') or '').lower():
            continue
        if not str(msg.get('content') or '').strip():
            continue

        timestamp = msg.get('created_at') or msg.get('created_at_str', '')

        if sid not in participants:
            participants[sid] = {
                'who': msg.get('who', sid),
                'last_spoke': timestamp,
                'message_count': 0,
            }

        participants[sid]['last_spoke'] = timestamp
        participants[sid]['message_count'] += 1
        last_speaker = sid

    return {
        'participants': participants,
        'last_speaker': last_speaker,
    }


async def fetch_transcript(game_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch and format the chat transcript"""
    logger.info("fetch_transcript: /silent on")
    await atlantis.client_command("/silent on")
    logger.info("fetch_transcript: /transcript chat")

    raw_transcript = await atlantis.client_command("/transcript chat")

    logger.info(f"fetch_transcript: received {len(raw_transcript)} entries")
    await atlantis.client_command("/silent off")
    logger.info("fetch_transcript: /silent off")

    if not raw_transcript:
        logger.error("!!! CRITICAL: rawTranscript is EMPTY - no messages received from client!")
        raise ValueError("Cannot process empty transcript")

    logger.info(f"rawTranscript has {len(raw_transcript)} entries before system message handling")

    if raw_transcript[0].get('role') == 'system':
        logger.info("Found system message in transcript - will use our own system prompt instead")

    transcript_dump_file = os.path.join(require_membership(game_key), 'raw_transcript.json')
    try:
        with open(transcript_dump_file, 'w') as f:
            json.dump(raw_transcript, f, indent=2, default=str)
        logger.info(f"Raw transcript written to {transcript_dump_file}")
    except Exception as e:
        logger.warning(f"Failed to write raw transcript to file: {e}")

    logger.info("=== FILTERING TRANSCRIPT ===")
    transcript: List[Dict[str, Any]] = []

    for i, msg in enumerate(raw_transcript):
        msg_type = msg.get('type')
        msg_sid = msg.get('sid')
        msg_role = msg.get('role')
        msg_content = str(msg.get('content', ''))[:50]
        logger.info(f"  [{i}] type={msg_type}, sid={msg_sid}, role={msg_role}, content={repr(msg_content)}...")

        if msg_type == 'chat':
            if msg_sid == 'system':
                logger.info(f"       -> SKIPPED (sid=system)")
                continue

            msg_who = str(msg.get('who', ''))
            if 'thinking' in msg_who.lower():
                logger.info(f"       -> SKIPPED (thinking entry, who={msg_who})")
                continue

            msg_content_full = msg.get('content', '')
            if not msg_content_full or not msg_content_full.strip():
                logger.info(f"       -> SKIPPED (blank content)")
                continue

            if 'data-metacol=' in msg_content_full or 'bot-table-cell' in msg_content_full:
                logger.warning(f"       -> SKIPPED (contains HTML table data, {len(msg_content_full)} chars)")
                continue

            MAX_ENTRY_SIZE = 4000
            if len(msg_content_full) > MAX_ENTRY_SIZE:
                logger.warning(f"       -> SKIPPED (oversized: {len(msg_content_full)} chars > {MAX_ENTRY_SIZE})")
                continue

            transcript.append({'role': 'user', 'content': [{'type': 'text', 'text': msg_content_full}]})
            logger.info(f"       -> INCLUDED as role=user (sid={msg_sid})")
        elif msg_type == 'description':
            msg_content_full = msg.get('content', '')
            if not msg_content_full or not str(msg_content_full).strip():
                logger.info(f"       -> SKIPPED (blank description)")
                continue
            transcript.append({'role': 'user', 'content': [{'type': 'text', 'text': f"[scene: {msg_content_full}]"}]})
            logger.info(f"       -> INCLUDED as description")
        else:
            logger.info(f"       -> SKIPPED (type != 'chat'|'description')")
    logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")

    return raw_transcript, transcript


async def chat(game_key: str):
    """Chat"""
    caller = atlantis.get_caller()
    if not caller:
        logger.warning("Chat fired without a caller identity")
        return

    raw_transcript, transcript = await fetch_transcript(game_key)
    logger.info(
        "Chat transcript fetched: %s raw entries, %s filtered entries",
        len(raw_transcript),
        len(transcript),
    )

    participants = analyze_participants(raw_transcript)
    speaker_sid = participants.get("last_speaker")
    if not speaker_sid:
        await atlantis.client_log("No chat speaker found in transcript")
        return

    location = None  # TODO: location resolution needs reimplementation
    if not location:
        await atlantis.client_log(f"\U0001f4cd {speaker_sid} has no position — nowhere to chat")
        return

    occupants = []  # TODO: occupant resolution needs reimplementation
    if not occupants:
        await atlantis.client_log(f"\U0001f4cd {speaker_sid} is alone in {location}")
        return

    names = []
    bots = []
    for ch in occupants:
        display = ch.get("displayName", ch["occupant"])
        names.append(display)
        # TODO: bot detection needs reimplementation

    await atlantis.client_log(
        f"\U0001f3e0 Room [{location}]: {', '.join(names)}"
    )

    if bots:
        next_up = bots[0]
        await atlantis.client_log(
            f"\U0001f3a4 Next to speak: {next_up.get('displayName', next_up['occupant'])}"
        )
    else:
        await atlantis.client_log("\U0001f3a4 No bots present")
