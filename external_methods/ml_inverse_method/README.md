# Toy ML Inverse Method

This is a small zip-upload test method for the KTC dashboard.

It follows the expected CLI contract:

```text
python main.py input_dir output_dir level
```

The method reads the selected `.mat` file from `input_dir` and writes a `.mat`
file containing a `reconstruction` array with shape `(256, 256)`.
