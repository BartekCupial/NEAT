python train.py \
    "task.name=slimevolley" \
    "task.max_steps=3000" \
    "neat.pop_size=300" \
    "neat.prob_add_node=0.05" \
    "neat.prob_add_connection=0.15" \
    "neat.compatibility_threshold=0.3" \
    "neat.survival_threshold=0.25" \
    "trainer.n_repeats=16" \
    "trainer.n_evaluations=100" \
    "trainer.max_iter=2000" \
    "trainer.test_interval=50" \
    "trainer.log_interval=10" \
    "trainer.backprop_steps=100" \
    "trainer.learning_rate=0.01" \
    "trainer.l2_penalty=0.001"