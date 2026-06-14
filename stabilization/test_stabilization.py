from stabilization_engine import stabilization_action

sample = {

    "vcsel_temp":45.4,
    "contrast":0.86,
    "optical_power":96,
    "cell_temp":80.5
}

result = stabilization_action(sample)

for r in result:
    print(r)