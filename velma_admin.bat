@echo off
:: Launches Velma CLI with Administrator privileges
powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k cd /d %~dp0 && velma_env\Scripts\activate && python cli.py'"
