#!/bin/bash
if ! [ -d "train0" ] || [ "$1" == "force" ];
then
    git clone https://github.com/circulosmeos/gdown.pl
    ./gdown.pl/gdown.pl https://drive.google.com/file/d/14z5mIh7Pwi5ywO5PS0BM5rBJgwA936PF/view train.zip

    # Unzip files
    echo Unzip train.zip
    unzip -o -q train.zip
else
    echo 'Data has been downloaded already!'
fi

if ! [ -f "data.pickle" ] || [ "$1" == "force" ];
then
    ./gdown.pl/gdown.pl https://drive.google.com/open?id=1R8RRQOP1EoBH2SPLj9foBhILCvCm4OpW data.pickle
else
    echo 'Data has been downloaded already!'
fi

ls
