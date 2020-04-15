var gulp = require('gulp'),
    uglify = require('gulp-uglify'),
    plumber = require('gulp-plumber'),
    browserSync = require('browser-sync');

var reload = browserSync.reload;
var exec = require('child_process').exec;

// Uglify javascript
// gulp.task('scripts', function() {
//   gulp.src('js/*.js')
//     .pipe(plumber())
//     .pipe(uglify())
//     .pipe(gulp.dest('build/js'))
// });

//Run Flask server
gulp.task('runserver', function() {
    var proc = exec('python songciGenerator/app.py');
});

// browser sync
gulp.task('browser-sync', function() {
  browserSync({
    notify: false,
    proxy: "127.0.0.1:5000"
  });

  gulp.watch([
      'songciGenerator/app/templates/*.*',
    'songciGenerator/app/static/css/*.*',
      'songciGenerator/app/static/js/*.*',
  ], reload);
})

// Default task: Watch Files For Changes & Reload browser
gulp.task('default', gulp.parallel('runserver', 'browser-sync'));