import pretty_midi
import uuid
import pygame


def get_duration_note(input):
    """
    :param input:
    :return duration, note, delay:
    """
    duration = []
    note = []
    delay = [0.3]
    for item in input:
        duration_list = item['duration']
        note_list = item['key']
        lyric = item['lyrics']
        for i in range(int((len(lyric) + 1)/2)):
            dur = duration_list[i]
            no = note_list[i]
            if ',' in no:
                note += no.split(',')
                l = len(no.split(','))
                duration += [float(dur)/l for _ in range(l)]
                delay += [0 for _ in range(l - 1)]
                delay.append(0.3)
            else:
                duration.append(dur)
                note.append(no)
                delay.append(0.3)
        delay[-1] += 0.7

    return (duration, note, delay)

def get_lyric_time(input):
    result = []
    last_time = 0
    for item in input:
        length = len(item['duration'])
        duration = length * 0.3 + 0.7
        for i in range(length):
            duration += float(item['duration'][i])
        result.append({'lyrics': item['lyrics'], 't': last_time})
        last_time += duration

    return result

def create_midi(input):
    durations, keys, delay = get_duration_note(input)
    mid = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    time = 0
    for i in range(len(durations)):
        time += delay[i]
        note = pretty_midi.Note(velocity=100, pitch=int(keys[i]), start=time, end=time + float(durations[i]))
        # lyric = pretty_midi.Lyric(text=lyrics[i], time=time)

        time += float(durations[i])
        piano.notes.append(note)
        # mid.lyrics.append(lyric)

    mid.instruments.append(piano)

    uuid_str = uuid.uuid4().hex
    ## here is the random name if necessary
    temp_file_name = 'tmpfile_%s.mid' % uuid_str
    mid.write('static/upload/music/' + temp_file_name)
    return temp_file_name
    # return 'output.mid'


