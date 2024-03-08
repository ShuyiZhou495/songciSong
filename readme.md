# 重要的是最后面的《需要的function》，别的不看也行
### 有兴趣的同学可以下载，不是必须的。以下的安装方法应该是对的，如果有同学试了不对可以来问我， 那我大概是记错了 ：）
## Demo
<img width="720" alt="page1" src="https://github.com/ShuyiZhou495/songci/assets/62908724/94a1e7eb-89f0-415f-b90e-148e537d4e31">
<img width="720" alt="page2" src="https://github.com/ShuyiZhou495/songci/assets/62908724/5ed01084-75e8-4bee-aaae-e14ccd1122aa">

## Server-side
0. 安装pipenv 用`pipenv -v`确定安装了
1. `pipenv install`
2. `pipenv shell`
3. `python app.py`

## front-end
只是为了用browser sync， 不想用的同学可以无视下面的
1. install node.js
2. check nodejs by `npm -v`
3. `npm install`
4. `gulp`

## 颜色
我用的都是这个网址的颜色：
[android material](https://material.io/design/color/#tools-for-picking-colors)

## 需要安装 musescore
remote 上用的是lilypond.
是用来输出图片的，这两个差不多一样，这个版本使用的是musescore。
如果安装musescore，在[fun_get_img.py](./fun_get_img.py) 的第92行，改成musescore的路径。
如果都装不了，可能会有报错。
如果想直接放弃生成图片，可以把[app.py](./app.py)的第39行改成`img_path=''`,这样啥也不用装。
