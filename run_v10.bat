@echo off
set PYTHONUNBUFFERED=1
"C:\Users\hyeon\miniconda3\envs\mosquito\python.exe" -u "d:\hyeon\공부\mosquitoes\train_v10.py" > "C:\Users\hyeon\AppData\Local\Temp\v10_out2.txt" 2> "C:\Users\hyeon\AppData\Local\Temp\v10_err2.txt"
echo Exit code: %errorlevel% >> "C:\Users\hyeon\AppData\Local\Temp\v10_out2.txt"
