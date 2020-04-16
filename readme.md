# 重要的是最后面的《需要的function》，别的不看也行
### 有兴趣的同学可以下载，不是必须的。以下的安装方法应该是对的，如果有同学试了不对可以来问我， 那我大概是记错了 ：）

## Server-side
0. 安装pipenv 用`pipenv -v`确定安装了
1. `pipenv install`
2. `pipenv shell`
3. `flask run`

## front-end
只是为了用browser sync， 不想用的同学可以无视下面的
1. install node.js
2. check nodejs by `npm -v`
3. `npm install`
4. `gulp`

## 颜色
我用的都是这个网址的颜色：
[android material](https://material.io/design/color/#tools-for-picking-colors)

## 需要的function
在[app.py](app.py) 里我写了这两个，应该是填充就行。。
觉得github麻烦的话直接把程序发我微信就行 ：）
```python
def get_song(lyrics):
    """
    :param lyrics:
    :return: the path of generated song： 'static/upload/music/'+<filename>
    """
    pass

def get_img(lyrics):
    """
    :param lyrics:
    :return: the path of generated 五线谱: ''static/upload/img/'+<filename>
    """
    pass
```