Separate UV environment was created specifically for the lab. It loses the benefit of contenerization with Docker but passing through either GPU or Apple Metal device works automatically significantly speeding up training time.
```
uv sync
uv run --with jupyter jupyter lab
```