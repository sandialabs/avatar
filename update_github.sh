#!/bin/bash

CWD=`pwd`
if [ $# -ne 2 ]; then
    echo "syntax: $0 <version tag> <github_clone_directory>"
    exit 1;
fi

VERSION=$1
GITHUBDIR=$2
TMPDIR=$(mktemp -d)

if [ -d $GITHUBDIR ]; then :
else echo "ERROR: Cannot find $GITHUBDIR"; exit -1; fi

# Purge emacs temp files
rm -f `find . -name \*~`

# Check the local repo
git diff --quiet
if [ $? -ne 0 ]; then echo "ERROR: local repo is not clean"; exit -1;fi
git diff --cached --quiet
if [ $? -ne 0 ]; then echo "ERROR: local repo is not clean"; exit -1;fi

# Copy files (except for the .git directory)
mv .git $TMPDIR
cp -r .  $GITHUBDIR
mv $TMPDIR/.git  .

# Do the git-fu on the gitlab end
cd $GITHUBDIR
git add .
if [ $? -ne 0 ]; then echo "ERROR: secondary git add failed"; exit -1; fi
git commit -m "Updating from main repo" .
if [ $? -ne 0 ]; then echo "ERROR: secondary git commit failed"; exit -1; fi
git tag $VERSION
if [ $? -ne 0 ]; then echo "ERROR: secondary git tag failed"; exit -1; fi
git push
if [ $? -ne 0 ]; then echo "ERROR: secondary git push failed"; exit -1; fi

# Tag this guy locally and push the tag
cd $CWD
git tag $VERSION
if [ $? -ne 0 ]; then echo "ERROR: local git tag failed"; exit -1; fi
git pull --rebase
if [ $? -ne 0 ]; then echo "ERROR: local git pull failed"; exit -1; fi
#git push
if [ $? -ne 0 ]; then echo "ERROR: local git push failed"; exit -1; fi


rm -rf $TMPDIR
