
import shlex
import sys
from optparse import OptionParser, SUPPRESS_HELP

from gomill import __version__
from gomill import ascii_boards
from gomill import gtp_controller
from gomill import gtp_games
from gomill.gtp_controller import GtpChannelError, BadGtpResponse
from gomill.common import format_vertex
import video_player

def print_move(colour, move, board, **kwargs):
    print(colour.upper(), format_vertex(move))

def print_board(colour, move, board, **kwargs):
    print( colour.upper(), format_vertex(move))
    print( ascii_boards.render_board(board))
    print()

def main():
##    usage = "%prog [options] --black='<command>' --white='<command>'"
##    parser = OptionParser(usage=usage)
##    parser.add_option("--black", help=SUPPRESS_HELP)
##    parser.add_option("--white", help=SUPPRESS_HELP)
##    parser.add_option("--komi", type="float", default=7.5)
##    parser.add_option("--size", type="int", default=19)
##    parser.add_option("--games", type="int", default=1, metavar="COUNT")
##    parser.add_option("--verbose", type="choice", choices=('0','1','2'),
##                      default=0, metavar="0|1|2")
##    parser.add_option("--sgfbase", type="string", metavar="FILENAME-PREFIX")
##
##    (options, args) = parser.parse_args()
##    if args:
##        parser.error("too many arguments")
##    if not options.black or not options.white:
##        parser.error("players not specified")
##
##    black_command = shlex.split(options.black)
##    white_command = shlex.split(options.white)
##    b_code = black_command[0]
##    w_code = white_command[0]
##    if b_code == w_code:
##        b_code += '-b'
##        w_code += '-w'
    #these used to be in an options object
    verbose = '2'
    size = 13
    komi = 5.5

    b_code = 'b'
    w_code = 'w'
    white_command = 'gnugo\gnugo --mode gtp --silent --color white'

    #video_source = 'tests/go game.mp4'
    video_source = 0

    game_controller = gtp_controller.Game_controller(b_code, w_code)
    try:
        player = video_player.VideoPlayer("camera_params.npz",
                                          video_source,
                                          size,
                                          "Gowatcher",
                                          rotate_pic=True,
                                          debug=True)
        engine = video_player.make_engine(player)
        channel = gtp_controller.Internal_gtp_channel(engine)
        black_controller = gtp_controller.Gtp_controller(channel, 'b')
        game_controller.set_player_controller('b', black_controller)
        game_controller.set_player_subprocess('w', white_command)
    except GtpChannelError as e:
        game_controller.close_players()
        sys.exit("error creating players:\n%s\n" % e)

    eds = game_controller.engine_descriptions
    if verbose == '1':
        print( 'Black: %s' % eds['b'].get_short_description())
        print( 'White: %s' % eds['w'].get_short_description())
    elif verbose == '2':
        print( 'Black: %s\n' % eds['b'].get_long_description())
        print( 'White: %s\n' % eds['w'].get_long_description())

    game = gtp_games.Gtp_game(
        game_controller,
        board_size=size,
        komi=komi,
        move_limit=1000)

    if verbose == '1':
        game.set_move_callback(print_move)
    elif verbose == '2':
        game.set_move_callback(print_board)
    game.allow_scorer('b')
    game.allow_scorer('w')

    try:
        game.prepare()
        game.run()
    except (GtpChannelError, BadGtpResponse) as e:
        game_controller.close_players()
        sys.exit("aborting run due to error:\n%s\n" % e)
    print(game.result.describe())

##        if options.sgfbase is not None:
##            sgf_game = game.make_sgf()
##            sgf_game.get_root().set("AP", ("Gomill twogtp", __version__))
##            try:
##                write_sgf(sgf_game, game_number, options.sgfbase)
##            except EnvironmentError, e:
##                sys.exit("error writing SGF file: %s" % e)

    game_controller.close_players()
    late_error_messages = game_controller.describe_late_errors()
    if late_error_messages:
        sys.exit(late_error_messages)

if __name__ == "__main__":
    main()
