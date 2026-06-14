from predict_lock import predict_lock

result = predict_lock(
{
    "vcsel_temp":45,
    "vcsel_current":3,
    "optical_power":100,
    "cell_temp":80,
    "magnetic_field":1.5,
    "contrast":0.95
}
)

print(result)