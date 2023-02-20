import tensorflow as tf

from ai_old import ai_player
from ai_arch import arch_a1_c0, arch_a1_c1, arch_a1_c2, arch_a1_d0, arch_a1_d1
from game import simulateGames
from board import testInitialBoard
old_model = tf.keras.models.load_model("models/evalBoard_0")
curr_model = tf.keras.models.load_model("models/amplify_0_arch_a1_c2_rate_96")

for arch in [arch_a1_d0, arch_a1_d1, arch_a1_c0, arch_a1_c1, arch_a1_c2]:
    print(arch.__name__, arch().summary())

exit()


print(old_model.summary())
print(curr_model.summary())

old_player = ai_player(old_model)
curr_player = ai_player(curr_model)

simulateGames(testInitialBoard, curr_player, old_player, 100, True)
