var gulp = require('gulp'),
    uglify = require('gulp-uglify'),
    plumber = require('gulp-plumber'),
    browserSync = require('browser-sync'),
    less = require('gulp-less');

var reload = browserSync.reload;
var exec = require('child_process').exec;

var files = [
      'templates/*.html',
    'static/css/*.css',
      'static/js/*.*',
    ];

// Uglify javascript
// gulp.task('scripts', function() {
//   gulp.src('js/*.js')
//     .pipe(plumber())
//     .pipe(uglify())
//     .pipe(gulp.dest('build/js'))
// });

// browser sync
gulp.task('browser-sync', function() {

    var proc = exec('flask run');
    browserSync({
        notify: false,
        proxy: "127.0.0.1:5000"
    });
});


gulp.task('less:watch', function () {
    gulp.watch(['./static/css/*.less']).on('change', function () {
        return gulp.src('./static/css/*.less')
        .pipe(less())
        .pipe(gulp.dest('./static/css'));
    });
});

gulp.task('watch', function () {
      gulp.watch(files).on('change', reload);
});

// Default task: Watch Files For Changes & Reload browser
gulp.task('default',gulp.parallel(['browser-sync', 'watch', 'less:watch']));