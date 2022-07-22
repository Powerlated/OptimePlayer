// @ts-check
const fs = require('fs');
const child_process = require('child_process');
const ffmpeg = require('fluent-ffmpeg');

const outDir = "playlist-out";

if (process.argv.length < 4) {
    console.log("Arguments: <path to DS ROM> <path to playlist>");
    process.exit(0);
}
if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir);
}

/** @param {string} string */
function removeLineBreaks(string) {
    return string.replace(/(\r\n|\n|\r)/gm, "");
}

let playlistStr = fs.readFileSync(process.argv[3]).toString();
let playlist = [];
let lines = playlistStr.split('\n');
for (let i in lines) {
    let line = lines[i];
    let lineSplit = line.split(' ');
    let sseqName = removeLineBreaks(lineSplit[0]);
    lineSplit.shift();
    let songName = removeLineBreaks(lineSplit.join(' '));
    console.log(`${sseqName}: ${songName}`);

    if (songName.length == 0) {
        console.error(`Error in line ${Number(i) + 1} of playlist`);
        continue;
    }

    playlist.push({ sseqName: sseqName, songName: songName });
}

let completeProcesses = 0;
let paths = [];
for (let i in playlist) {
    let entry = playlist[i];
    let path = `${outDir}/${Number(i) + 1}. ${entry.songName}`;
    paths.push(path);
    let childProc = child_process.fork("./video-exporter.js", [
        process.argv[2],
        entry.sseqName,
        path,
    ],
        { env: { songName: entry.songName, nextSongName: playlist[Number(i) + 1]?.songName } }
    );

    childProc.on('exit', () => {
        completeProcesses++;

        if (completeProcesses == playlist.length) {
            let f = ffmpeg();
            for (let i of paths) {
                f.addInput(`${i}.mp4`);
            }
            f.on('start', (cmdline) => console.log(cmdline));
            f.outputOptions('-crf 30');
            f.audioCodec('aac');
            f.audioBitrate('264k');
            // @ts-ignore
            f.mergeToFile(`${outDir}/Playlist.mp4`, outDir);
        }
    });
}