"""Owner setup for the simulation command-center chat using the existing Chat engine."""
import os
from urllib.parse import urlencode, urlsplit
import atlantis
from .game import _game_rows, _game_read, _game_set_state
from .runner import game_new
from .roster import roster_create, roster_set_slot, roster_spawn
from .camera import camera_bind


@visible
async def command_center(public_base_url: str) -> dict:
    """Create or reconnect the owner's command-center chat and return its login-protected HTTPS link. Uses Chat discovery and the provider key sourced before MCP startup; does not reset the simulation."""
    if atlantis.get_caller() not in atlantis.get_owner_usernames():
        raise PermissionError('Authenticated host owner required')
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise RuntimeError('Source OPENROUTER_API_KEY before starting the MCP server')
    parts=urlsplit(public_base_url)
    if parts.scheme!='https' or not parts.netloc or parts.query or parts.fragment or parts.path not in ('','/'):
        raise ValueError('public_base_url must be an HTTPS origin')
    window=atlantis.get_user_game_id();session=atlantis.get_session_key()
    matches=[g for g in _game_rows() if str(g.get('user_game_id'))==str(window) and session in (_game_read(g['game_key']).get('members') or {})]
    if len(matches)>1:
        raise RuntimeError('Multiple Chat games are bound to this session')
    if matches:
        game_key=matches[0]['game_key']
        if matches[0].get('roster_scene') not in (None,'command_center'):
            raise RuntimeError('This window already belongs to another Chat scene; open a separate game window')
    else:
        game_key=(await game_new())['game_key']
    if not matches or not matches[0].get('roster_scene'):
        await roster_create(game_key,'command_center')
        await roster_set_slot(game_key,'Operator','human',atlantis.get_caller())
        await roster_spawn(game_key,'Operator','SecurityOffice')
        await roster_spawn(game_key,'Command','SecurityOffice')
    await atlantis.client_command('/cursor join',{'game_key':game_key})
    await camera_bind(game_key,'SecurityOffice')
    folder=atlantis.get_script_folder()
    await atlantis.client_command('/cd '+folder,shell='caller')
    await atlantis.client_command('/callback set chat '+folder+'/chat_callback',shell='caller')
    await atlantis.client_command('/callback set preflight '+folder+'/preflight_callback',shell='caller')
    await _game_set_state(game_key,'running')
    from .chat_callback import greet_entrant
    await greet_entrant(game_key,atlantis.get_caller(),'SecurityOffice')
    await atlantis.client_command('/terminal off',shell='caller')
    query=urlencode({'game':window,'sid':atlantis.get_caller(),'shell':atlantis.get_caller_shell_path()})
    return {'chatUrl':public_base_url.rstrip('/')+'/chat.html?'+query,'persona':'Arnold','location':'SecurityOffice','gameKey':game_key,'tools':'discovered during chat turns','providerKeyLoaded':True}
