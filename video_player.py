from gomill import gtp_engine
from gomill import gtp_states
import gowatcher


class VideoPlayer(object):
    """Player for use with gtp_state."""

    def __init__(self, cam_data_file, source = 0,
                 board_size = 19, window_name = None,
                 rotate_pic = True, debug = False):
        self.watcher = gowatcher.GoWatcher(source, board_size,
                                           cam_data_file, window_name,
                                           rotate_pic,
                                           debug)
        self.watcher.initialize()

    def genmove(self, game_state, player):
        """Move generator that chooses a random empty point.
        game_state -- gtp_states.Game_state
        player     -- 'b' or 'w'
        This may return a self-capture move.
        """
        return self.watcher.genmove(game_state, player)

    def handle_name(self, args):
        return "GoWatcher player"

    def handle_version(self, args):
        return ""


    def get_handlers(self):
        return {
            'name'            : self.handle_name,
            'version'         : self.handle_version,
            }

def make_engine(player):
    """Return a Gtp_engine_protocol which runs the specified player."""

    gtp_state = gtp_states.Gtp_state(
        move_generator=player.genmove,
        acceptable_sizes=(9, 13, 19))
    engine = gtp_engine.Gtp_engine_protocol()
    engine.add_protocol_commands()
    engine.add_commands(gtp_state.get_handlers())
    engine.add_commands(player.get_handlers())
    return engine
