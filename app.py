import math
import re
from collections import Counter
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
TOKEN_RE = re.compile(r"\b[\w’'-]+\b", re.UNICODE)
STATE = {"trained": False, "corpus": "", "tokens": [], "vocab": set(), "unigram": Counter(), "ngrams": {}, "context_totals": {}}

def tokenize(text): return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]

def train_model(corpus):
    corpus = corpus.strip()
    if not corpus: raise ValueError("Please provide a corpus before training the model.")
    tokens = tokenize(corpus)
    if len(tokens) < 8: raise ValueError("Please provide a larger corpus (at least 8 words).")
    ngrams, context_totals, unigram = {}, {}, Counter(tokens)
    for n in range(1, 6):
        c = Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)); ngrams[n] = c
    for n in range(2,6):
        context_totals[n] = Counter({})
        for gram,count in ngrams[n].items(): context_totals[n][gram[:-1]] += count
    STATE.update({"trained":True,"corpus":corpus,"tokens":tokens,"vocab":set(tokens),"unigram":unigram,"ngrams":ngrams,"context_totals":context_totals})

def probability(history, word, k=0.5):
    history=tuple(x.lower() for x in history); word=word.lower(); V=max(len(STATE["vocab"]),1)
    for order in range(min(5,len(history)+1),1,-1):
        h=history[-(order-1) :]; context=STATE["context_totals"][order].get(h,0)
        if context:
            return (STATE["ngrams"][order].get(h+(word,),0)+k)/(context+k*V)
    total=sum(STATE["unigram"].values())
    return (STATE["unigram"].get(word,0)+k)/(total+k*V)

def rank_next_words(history, top_k=5):
    candidates=[]
    for word in STATE["vocab"]:
        s=math.log(max(probability(history,word),1e-12))+0.03*math.log1p(STATE["unigram"].get(word,0)); candidates.append((s,word))
    candidates.sort(reverse=True); top=candidates[:top_k]
    if not top:return []
    mx=top[0][0]; ex=[math.exp(s-mx) for s,_ in top]; den=sum(ex) or 1
    return [{"word":w,"rank_score":round(s,6),"relative_probability":round(e/den*100,1)} for (s,w),e in zip(top,ex)]

def levenshtein(a,b):
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a.lower(),1):
        cur=[i]
        for j,cb in enumerate(b.lower(),1): cur.append(min(cur[j-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def spelling_candidates(word, history, limit=5):
    word=word.lower(); md=1 if len(word)<=4 else 2 if len(word)<=7 else 3; out=[]
    for cand in STATE["vocab"]:
        if abs(len(cand)-len(word))>md: continue
        d=levenshtein(word,cand)
        if d<=md:
            score=-1.6*d+math.log(max(probability(history,cand),1e-12))+0.03*math.log1p(STATE["unigram"].get(cand,0)); out.append((score,cand,d))
    out.sort(reverse=True)
    return [{"word":c,"edit_distance":d,"score":round(s,5)} for s,c,d in out[:limit]]

def bigram_rows(tokens):
    rows=[]
    for a,b in zip(tokens,tokens[1:]):
        ctx=STATE["context_totals"][2].get((a,),0)
        rows.append({"bigram":f"{a} {b}","no_smoothing":round(STATE["ngrams"][2].get((a,b),0)/ctx,6) if ctx else 0.0,"add_k":round(probability([a],b,.5),6),"laplace":round(probability([a],b,1.0),6)})
    return rows

def perplexity(tokens):
    if len(tokens)<2:return None
    s=0; n=0
    for i in range(1,len(tokens)):
        s += math.log(max(probability(tokens[max(0,i-4):i],tokens[i],.5),1e-12)); n += 1
    return math.exp(-s/n)

def reference_similarity(tokens):
    if len(tokens)<3:return {"overlap_percent":0.0,"matched_trigrams":0,"total_trigrams":0,"phrases":[],"level":"none","label":"No significant textual overlap detected"}
    test=set(zip(tokens,tokens[1:],tokens[2:])); ref=STATE["tokens"]; refset=set(zip(ref,ref[1:],ref[2:])); matched=test&refset; total=len(test); pct=len(matched)/total*100 if total else 0
    if pct>=70: level,label="very_high","Plagiarism / exact-phrase overlap detected"
    elif pct>=40: level,label="high","High textual overlap detected"
    elif pct>=20: level,label="moderate","Moderate textual overlap detected"
    else: level,label="low","No significant textual overlap detected"
    return {"overlap_percent":round(pct,1),"matched_trigrams":len(matched),"total_trigrams":total,"phrases":sorted(" ".join(g) for g in matched)[:20],"level":level,"label":label}

def conclusions(rows,sim,ppl):
    out=[]; zero=sum(r["no_smoothing"]==0 for r in rows)
    if zero: out.append(f"No-smoothing assigned probability 0 to {zero} transition(s) because the exact bigram was not present in the corpus. Add-k and Laplace avoid this zero-probability problem.")
    elif rows:
        vals=[sum(r[k] for r in rows)/len(rows) for k in ("no_smoothing","add_k","laplace")]
        out.append(f"Average bigram probabilities were {vals[0]:.4f} (no smoothing), {vals[1]:.4f} (Add-k), and {vals[2]:.4f} (Laplace). Smoothing redistributes probability mass and makes the model less brittle for unseen combinations.")
    if sim["level"]=="very_high": out.append(f"{sim['overlap_percent']}% of the sentence's distinct trigrams overlap with the corpus. Strong textual similarity was found, so the system flags plagiarism/exact-phrase overlap. This is an indicator, not proof by itself.")
    elif sim["level"] in {"high","moderate"}: out.append(f"{sim['overlap_percent']}% n-gram overlap was found with the corpus, indicating similar wording that should be reviewed.")
    else: out.append("Very little n-gram overlap was found with the corpus, so no significant textual similarity was detected.")
    if ppl is not None: out.append(f"Perplexity is {ppl:.2f}. Lower perplexity means the trained model finds the sentence more predictable; higher perplexity means greater uncertainty.")
    else: out.append("The sentence is too short to calculate perplexity.")
    return out

@app.route('/')
def index(): return render_template('index.html')
@app.route('/api/health')
def health(): return jsonify({'trained':STATE['trained'],'tokens':len(STATE['tokens']),'vocabulary':len(STATE['vocab'])})
@app.route('/api/train',methods=['POST'])
def train():
    data=request.get_json(silent=True) or {}
    try: train_model(data.get('corpus',''))
    except ValueError as e: return jsonify({'ok':False,'error':str(e)}),400
    return jsonify({'ok':True,'tokens':len(STATE['tokens']),'vocabulary':len(STATE['vocab']),'ngram_counts':{str(n):len(STATE['ngrams'][n]) for n in range(1,6)}})
@app.route('/api/predict',methods=['POST'])
def predict():
    if not STATE['trained']: return jsonify({'ok':False,'error':'Train a model from your corpus first.'}),400
    text=(request.get_json(silent=True) or {}).get('text',''); toks=tokenize(text)
    if not toks:return jsonify({'ok':True,'predictions':[],'corrections':[]})
    last=toks[-1]; return jsonify({'ok':True,'predictions':rank_next_words(toks,5),'corrections':spelling_candidates(last,toks[:-1],5) if last not in STATE['vocab'] else []})
@app.route('/api/analyze',methods=['POST'])
def analyze():
    if not STATE['trained']: return jsonify({'ok':False,'error':'Train a model from your corpus first.'}),400
    text=(request.get_json(silent=True) or {}).get('text',''); toks=tokenize(text)
    if not toks:return jsonify({'ok':False,'error':'Enter a sentence to analyse.'}),400
    rows=bigram_rows(toks); ppl=perplexity(toks); sim=reference_similarity(toks)
    return jsonify({'ok':True,'probability_table':rows,'perplexity':round(ppl,3) if ppl is not None else None,'similarity':sim,'conclusions':conclusions(rows,sim,ppl)})
if __name__=='__main__': app.run(debug=True)
