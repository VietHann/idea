import wave, math, random, struct
from pathlib import Path
P=Path('Resources/Audio'); P.mkdir(parents=True,exist_ok=True); SR=22050

def write(name,samples):
    with wave.open(str(P/name),'w') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(b''.join(struct.pack('<h',max(-32767,min(32767,int(s*32767)))) for s in samples))
def tone(freq,dur,vol=.35,decay=True,slide=0):
    out=[]; n=int(SR*dur)
    for i in range(n):
        t=i/SR; f=freq+slide*t/dur; env=(1-t/dur) if decay else 1; out.append(math.sin(2*math.pi*f*t)*vol*env)
    return out
def mix(parts):
    n=max(map(len,parts)); out=[0.0]*n
    for p in parts:
        for i,v in enumerate(p): out[i]+=v
    return [max(-1,min(1,v)) for v in out]
write('gulp.wav',mix([tone(280,.18,.3,slide=-100),tone(140,.22,.25,slide=-50)])); write('pop.wav',mix([tone(620,.11,.28,slide=260),tone(900,.07,.12,slide=-120)])); write('bonk.wav',mix([tone(110,.18,.42,slide=-40),tone(70,.25,.22)])); write('power.wav',mix([tone(420,.35,.18,slide=600),tone(210,.4,.16,slide=400)])); write('win.wav',tone(523,.18,.18,False)+tone(659,.18,.18,False)+tone(784,.24,.2,False)+tone(1046,.45,.22,True))
bpm=136; beat=60/bpm; step=beat/2; notes=[261.6,329.6,392,523.3,392,329.6,293.7,349.2,440,587.3,440,349.2,329.6,392,493.9,659.3]; bass=[130.8,130.8,146.8,146.8,164.8,164.8,146.8,146.8]; total=int(SR*step*16*16/4); out=[0.0]*total
for k in range(64):
    for j,n in enumerate(notes):
        start=int((k*4+j)*step*SR)
        if start>=total: break
        dur=int(step*.78*SR)
        for i in range(min(dur,total-start)):
            t=i/SR; env=(1-i/dur)**1.8; out[start+i]+=0.07*math.sin(2*math.pi*n*t)*env+0.025*math.sin(4*math.pi*n*t)*env
    b=bass[k%len(bass)]; start=int(k*4*step*SR); dur=int(4*step*.9*SR)
    for i in range(min(dur,total-start)):
        t=i/SR; out[start+i]+=0.07*math.sin(2*math.pi*b*t)*(1-i/dur)
for s in range(int(total/(step*SR))):
    start=int(s*step*SR); dur=int(.035*SR)
    for i in range(min(dur,total-start)): out[start+i]+=random.uniform(-1,1)*0.025*(1-i/dur)
write('blob_theme.wav',[max(-.85,min(.85,x)) for x in out]); print('audio generated')
