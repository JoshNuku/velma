@echo off
title Velma
cd /d %~dp0
call velma_env\Scripts\activate
python cli.py
pause
