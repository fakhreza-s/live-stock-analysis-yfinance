@echo off
set /p commit_msg=Enter commit message: 

echo Committing files to local repository...
git add .
git commit -m "%commit_msg%"

echo Pushing changes to GitHub...
git push -u origin master

echo Done!
pause