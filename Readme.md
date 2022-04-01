图片标注工具:https://github.com/tzutalin/labelImg

brew install python3
pip3 install pipenv
pipenv run pip install pyqt5==5.15.2 lxml
pipenv run make qt5py3
pipenv run python3 labelImg.py
[可选] rm -rf build dist ; python setup.py py2app -A ; mv " dist/labelImg.app " /Applications
