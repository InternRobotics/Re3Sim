## eval the policy in simulation

1. generate test cases and modify the yaml file:

```bash
python scripts/generate_test_cases.py
```

2. run the policy in simulation:

```bash
python standalone/example/eval_policy_for_multi_item.py
```

> We also provide a trained policy for the `pick a bottle` task, you can download it from [here](https://huggingface.co/RE3SIM/act-models). And you can run the policy in simulation by modifying the yaml file and running the command above.