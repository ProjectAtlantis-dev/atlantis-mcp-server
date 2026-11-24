"""Chat — transcript fetching and participant analysis."""

# TODO(Bot split, in progress): bots are moving out of Chat/Game/Bots into a
# top-level Bot app that may run on a different machine, so Chat must reach it
# via tool calls, never imports. Bot holds static bot info and (later) manages
# bot tool lists. This machine stays game master and runs the LLM turn. The game
# owns locations: legal locations and entry locations belong to roles, not bots.

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
        # An error the user raised by typing a command calls the callback too (Node
        # names its typedCommandId), so that user is the one the bots now answer
        if msg.get('type') == 'error' and 'typedCommandId' in msg:
            last_speaker = msg['sid']
            continue
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


async def fetch_transcript(game_name: str, game_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Fetch and format the chat transcript"""
    # Pull by the caller's terminal: Node returns the chat shell it is bound to
    # (by the folder's @enter, never the whole game) plus that terminal's own
    # commands and errors, which this user saw and nobody else did
    caller_shell = atlantis.get_caller_shell_path()
    if not caller_shell:
        raise RuntimeError("fetch_transcript: no caller shell; it must run from the user's terminal")

    logger.info("fetch_transcript: /silent on")
    await atlantis.client_command("/silent on")
    logger.info(f"fetch_transcript: /transcript get {caller_shell}")

    raw_transcript = await atlantis.client_command(f"/transcript get {json.dumps(caller_shell)}")

    logger.info(f"fetch_transcript: received {len(raw_transcript)} entries")
    await atlantis.client_command("/silent off")
    logger.info("fetch_transcript: /silent off")

    if not raw_transcript:
        logger.error("!!! CRITICAL: rawTranscript is EMPTY - no messages received from client!")
        raise ValueError("Cannot process empty transcript")

    logger.info(f"rawTranscript has {len(raw_transcript)} entries before system message handling")

    if raw_transcript[0].get('role') == 'system':
        logger.info("Found system message in transcript - will use our own system prompt instead")

    transcript_dump_file = os.path.join(require_membership(game_name, game_key), 'raw_transcript.json')
    try:
        with open(transcript_dump_file, 'w') as f:
            json.dump(raw_transcript, f, indent=2, default=str)
        logger.info(f"Raw transcript written to {transcript_dump_file}")
    except Exception:
        logger.exception(f"Failed to write raw transcript to {transcript_dump_file}")

    logger.info("=== FILTERING TRANSCRIPT ===")
    transcript: List[Dict[str, Any]] = []
    # What the user typed in the caller's terminal, by event id: Node gives each
    # error the typedCommandId of the command it came from, and that command
    # comes earlier in the transcript
    typed_commands: Dict[int, str] = {}

    def typed_command_of(msg: Dict[str, Any]) -> str:
        typed_id = msg['typedCommandId']
        if typed_id not in typed_commands:
            raise ValueError(f"Transcript event {msg['id']} ran under typed command {typed_id}, which the transcript does not hold before it")
        return typed_commands[typed_id]

    for i, msg in enumerate(raw_transcript):
        msg_type = msg.get('type')
        msg_sid = msg.get('sid')
        msg_role = msg.get('role')
        msg_content = str(msg.get('content', ''))[:50]
        logger.info(f"  [{i}] type={msg_type}, sid={msg_sid}, role={msg_role}, content={repr(msg_content)}...")

        if msg_type == 'command':
            # Typed by the user in the caller's terminal (see Node getTranscriptItems)
            typed_commands[msg['id']] = str(msg.get('content', ''))
            transcript.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': f"[the user typed in their terminal: {typed_commands[msg['id']]}]"}],
                '_speaker_sid': msg_sid,
                '_event_id': str(msg['id']),
            })
            logger.info(f"       -> INCLUDED as typed command")
        elif msg_type == 'chat':
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

            transcript.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': msg_content_full}],
                '_speaker_sid': msg_sid,
                '_event_id': str(msg['id']),
            })
            logger.info(f"       -> INCLUDED as role=user (sid={msg_sid})")
        elif msg_type == 'description':
            msg_content_full = msg.get('content', '')
            if not msg_content_full or not str(msg_content_full).strip():
                logger.info(f"       -> SKIPPED (blank description)")
                continue
            transcript.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': f"[scene: {msg_content_full}]"}],
                '_event_id': str(msg['id']),
            })
            logger.info(f"       -> INCLUDED as description")
        elif msg_type == 'error':
            # A failure in the caller's terminal (see Node getTranscriptItems):
            # the user saw it, so the bot must too
            msg_content_full = msg.get('content', '')
            if not msg_content_full or not str(msg_content_full).strip():
                logger.info(f"       -> SKIPPED (blank error)")
                continue
            # Node names the typed command that raised it, unless the error came
            # before any command was accepted
            if 'typedCommandId' in msg:
                error_text = f"[error in the user's terminal from {typed_command_of(msg)!r}: {msg_content_full}]"
            else:
                error_text = f"[error in the user's terminal: {msg_content_full}]"
            transcript.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': error_text}],
                '_speaker_sid': msg_sid,
                '_event_id': str(msg['id']),
            })
            logger.info(f"       -> INCLUDED as error")
        else:
            logger.info(f"       -> SKIPPED (type != 'command'|'chat'|'description'|'error')")
    logger.info(f"=== END FILTERING: {len(transcript)} messages included ===")

    filtered_dump_file = os.path.join(require_membership(game_name, game_key), 'filtered_transcript.json')
    try:
        with open(filtered_dump_file, 'w') as f:
            json.dump(transcript, f, indent=2, default=str)
        logger.info(f"Filtered transcript written to {filtered_dump_file}")
    except Exception:
        logger.exception(f"Failed to write filtered transcript to {filtered_dump_file}")

    return raw_transcript, transcript

