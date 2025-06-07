python train.py \
    "task.name=spiral" \
    "neat.pop_size=150" \
    "neat.prob_add_node=0.05" \
    "neat.prob_add_connection=0.15" \
    "neat.compatibility_threshold=3.0" \
    "neat.activation_function=[relu,square,sin,cos]" \
    "trainer.max_iter=100" \
    "trainer.backprop_steps=1000" \
    "trainer.learning_rate=0.03" \
    "trainer.l2_penalty=0.000"
