#! /usr/bin/env bash
now="$(date +'%Y/%m/%d')";
git add .; git commit -m "Update $now"; git push;
