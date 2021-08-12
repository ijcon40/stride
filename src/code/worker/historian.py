"""
Holds the worker which trains the chess model using self play data.
"""
import glob
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time

import chess.pgn

from src.code.agent.model_chess import ChessModel
from src.code.agent.player_chess import ChessPlayer
from src.code.config import Config
from src.code.env.chess_env import ChessEnv, Winner
from src.code.lib.data_helper import get_game_data_filenames, write_game_data_to_file, pretty_print
from src.code.lib.model_helper import load_best_model_weight, save_as_best_model, \
    reload_best_model_weight_if_changed
from tqdm import tqdm

logger = getLogger(__name__)


def start(config: Config):
    # take our pgns and parse them into games
    pgn_files = glob.glob('F:\\Lichess Elite Database\\Lichess Elite Database\\*.pgn')
    print('Historian: loading games')
    games = []
    games_to_load=1000
    for f in tqdm(pgn_files):
        if len(games) > games_to_load:
            break
        # load 100 thousand games
        with open(f, 'r') as file:
            print(f'Historian: loaded {len(games)} games')
            game = chess.pgn.read_game(file)
            while game is not None and len(games) <= games_to_load:
                games.append(game)
                game = chess.pgn.read_game(file)
    print(f'Historian: completed loading {len(games)} games from pgns')
    historian = HistorianWorker(config)
    historian.set_history(games)
    return historian.start(games)


# noinspection PyAttributeOutsideInit
class HistorianWorker:
    """
    Worker which trains a chess model using self play data. ALl it does is do self play and then write the
    game data to file, to be trained on by the optimize worker.

    Attributes:
        :ivar Config config: config to use to configure this worker
        :ivar ChessModel current_model: model to use for self play
        :ivar Manager m: the manager to use to coordinate between other workers
        :ivar list(Connection) cur_pipes: pipes to send observations to and get back mode predictions.
        :ivar list((str,list(float))): list of all the moves. Each tuple has the observation in FEN format and
            then the list of prior probabilities for each action, given by the visit count of each of the states
            reached by the action (actions indexed according to how they are ordered in the uci move list).
    """

    def __init__(self, config: Config):
        self.config = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.config.play.search_threads) for _ in
                                      range(self.config.play.max_processes)])
        self.buffer = []

    def set_history(self, games):
        self.games = games
        self.game_idx = 0

    def start(self, pgns=None):
        """
        Do self play and write the data to the appropriate file.
        """
        self.buffer = []

        futures = deque()
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            for game_idx in range(self.config.play.max_processes * 2):
                futures.append(
                    executor.submit(self_play_buffer, self.config, cur=self.cur_pipes, game=self.games[self.game_idx]))
            game_idx = 0
            while True and game_idx<len(self.games) and not len(futures) == 0:
                game_idx += 1
                start_time = time()
                env, data = futures.popleft().result()
                print(f"game {game_idx:3} time={time() - start_time:5.1f}s "
                      f"halfmoves={env.num_halfmoves:3} {env.winner:12} "
                      f"{'by resign ' if env.resigned else '          '}")

                pretty_print(env, ("current_model", "current_model"))
                self.buffer += data
                self.game_idx += 1
                if (game_idx % self.config.play_data.nb_game_in_file) == 0:
                    self.flush_buffer()
                    reload_best_model_weight_if_changed(self.current_model)
                #futures.append(executor.submit(self_play_buffer, self.config, cur=self.cur_pipes, game=self.games[self.game_idx]))  # Keep it going

    def load_model(self):
        """
        Load the current best model
        :return ChessModel: current best model
        """
        model = ChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            model.build()
            save_as_best_model(model)
        return model

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def remove_play_data(self):
        """
        Delete the play data from disk
        """
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        for i in range(len(files) - self.config.play_data.max_file_num):
            os.remove(files[i])


def self_play_buffer(config, cur, game) -> (ChessEnv, list):
    """
    Play one game and add the play data to the buffer
    :param Config config: config for how to play
    :param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
        predictions. One will be removed from this list during the game, then added back
    :param Game game: pgn game to recreate
    :return (ChessEnv,list((str,list(float)): a tuple containing the final ChessEnv state and then a list
        of data to be appended to the SelfPlayWorker.buffer
    """
    pipes = cur.pop()  # borrow
    env = ChessEnv().reset()
    # we get the moves from the game state

    white = ChessPlayer(config, pipes=pipes)
    black = ChessPlayer(config, pipes=pipes)

    # here we don't even let white or black have any choice
    # they are playing tournament moves and learning
    action_idx = 0
    actions = [move for move in game.mainline_moves()]

    while not env.done and action_idx<len(actions):
            if env.white_to_move:
                action = white.sl_action(env.observation, actions[action_idx].uci(), weight=.3)  # ignore=True
            else:
                action = black.sl_action(env.observation, actions[action_idx].uci(), weight=.3)  # ignore=True
            env.step(action, True)
            action_idx += 1
    # should be able to set winner if applicable, notably from pgn
    # figure out the winner from the game
    result = game.headers["Result"]
    env.result=result
    if not env.board.is_game_over() and result != '1/2-1/2':
        env.resigned = True
    if result == '1-0':
        env.winner = Winner.white
        black_win = -1
    elif result == '0-1':
        env.winner = Winner.black
        black_win = 1
    else:
        env.winner = Winner.draw
        black_win = 0

    black.finish_game(black_win)
    white.finish_game(-black_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])

    cur.append(pipes)
    return env, data
