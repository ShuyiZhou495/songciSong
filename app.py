## music player reference: https://github.com/Zegendary/Garbage/tree/master/%E5%A4%A7%E8%AE%BE%E8%AE%A1/simple%20music%20player/Random%20music%20player

from flask import Flask, redirect, url_for, request, render_template
from flask_bootstrap import Bootstrap
from flask_fontawesome import FontAwesome

app = Flask(__name__)
bootstrap = Bootstrap(app)
fa = FontAwesome(app)

def get_song(lyrics):
    """
    :param lyrics:
    :return: the path of generated song： eg, 'static/upload/music/output_song1.wav'
    """
    pass

def get_img(lyrics):
    """
    :param lyrics:
    :return: the path of generated 五线谱: eg, 'static/upload/img/output_img1.jpg'
    """
    pass

@app.route('/to_song')
def to_song():
    lyrics = request.args.get('lyric')
    music_path = get_song(lyrics)
    img_path = get_img(lyrics)
    return render_template('2song.html',
                           music_path="static/upload/music/output_song1.wav",
                           lyric=lyrics,
                           img_path='static/upload/img/output_img1.jpg')

@app.route('/')
def index():
    return render_template('index.html')


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404



if __name__ == '__main__':
    app.run()
