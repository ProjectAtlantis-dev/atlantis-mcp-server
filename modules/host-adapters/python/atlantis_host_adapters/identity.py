"""Shared operator-approved identity bindings; never infer an account from SID.

The host authenticates calls according to its own trust boundary. This module
maps that context to an existing immutable account identifier and scenario.
The policy file is trusted server configuration, not a dynamic-function argument.
"""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from uuid import UUID


@dataclass(frozen=True)
class Principal:
    caller: str
    user_game_id: int
    external_user_id: str
    scenario: str
    permissions: frozenset[str]


def resolve_principal(context, permission: str, *, policy_path=None) -> Principal:
    if context is None or not context.caller_sid or type(context.user_game_id) is not int or context.user_game_id <= 0:
        raise PermissionError('Authenticated caller and positive host game ID are required')
    path = policy_path or os.environ.get('ATLANTIS_IDENTITY_BINDINGS')
    if not path:
        raise PermissionError('ATLANTIS_IDENTITY_BINDINGS is required; SID identity fallback is forbidden')
    policy = json.loads(Path(path).read_text())
    version = policy.get('version')
    if version not in (1, 2) or not isinstance(policy.get('bindings'), list):
        raise ValueError('Unsupported identity bindings policy')
    seen = set()
    selected = None
    for item in policy['bindings']:
        caller, game = item.get('caller'), item.get('user_game_id')
        external, scenario = item.get('external_user_id'), item.get('scenario')
        permissions = item.get('permissions')
        if version == 2:
            subject = item.get('subject_uuid')
            if not isinstance(subject, str) or str(UUID(subject)) != subject:
                raise ValueError('Canonical server-approved subject UUID required')
            external = 'atlantis-subject:' + subject
        if (not isinstance(caller, str) or not caller.strip()
                or (version == 1 and (type(game) is not int or game <= 0
                    or not isinstance(external, str) or not re.fullmatch(r'x_user:[1-9][0-9]*', external)))
                or not isinstance(scenario, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', scenario)
                or not isinstance(permissions, list) or any(p not in ('simulation', 'bank', 'world') for p in permissions)):
            raise ValueError('Invalid identity binding; verified numeric x_user and explicit permissions required')
        key = (caller, game) if version == 1 else caller
        if key in seen:
            raise ValueError('Duplicate caller/game identity binding')
        seen.add(key)
        if key == ((context.caller_sid, context.user_game_id) if version == 1 else context.caller_sid):
            selected = Principal(caller, context.user_game_id, external, scenario, frozenset(permissions))
    if selected is None or permission not in selected.permissions:
        raise PermissionError('Caller/game has no approved binding for this module')
    return selected


def current_principal(permission='simulation') -> Principal:
    import atlantis
    return resolve_principal(atlantis.get_context(), permission)


def authorized_scenario(requested: str) -> str:
    principal = current_principal()
    if requested != principal.scenario:
        raise PermissionError('Requested scenario is outside the current host game binding')
    return principal.scenario


def bank_identity(permission: str) -> dict:
    principal = current_principal(permission)
    return {'externalUserId': principal.external_user_id, 'displayName': principal.caller,
            'identityAssurance': 'operator-bound-x_user' if principal.external_user_id.startswith('x_user:')
            else 'operator-bound-subject-uuid'}
