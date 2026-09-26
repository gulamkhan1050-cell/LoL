"""
lol_trainer.py  -  League of Legends pro-match predictor (run on your PC)

Data: PandaScore free plan (past matches, game winners and lengths, tournament rosters, tiers).
      Sign up at pandascore.co, copy your token, paste it below or set PANDASCORE_TOKEN.

Setup:   pip install requests numpy
Run:     python lol_trainer.py                 # ~12 months
         python lol_trainer.py --months 18
         python lol_trainer.py --demo          # synthetic data, offline pipeline check

Raw data is cached in cache_lol/ so re-runs only download what is new.
"""

import argparse, json, math, os, random, re, sys, time
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

import numpy as np
import requests

# ------------------------------------------------------------------ config
TOKEN ="di0FFBr1_1GjEpPMJNxCkjdM2cHqPMkUQuVLBZUOITRfjsUK8pc"
# Set PANDASCORE_TOKEN as an environment variable, or on GitHub as a repository secret.
# Do NOT paste the token on the line above: this file goes into a public repo, and anything
# committed there is public forever even after you delete it.
BASE = 1500.0
K_TEAM = 32.0
K_PLAYER = 20.0
TIER_BASE = {1: 1500.0, 2: 1480.0, 3: 1400.0, 4: 1330.0, 5: 1280.0}   # entry rating by tier of a team's first event: regional pools start lower
TIER_K = {1: 1.3, 2: 1.15, 3: 1.0, 4: 0.85, 5: 0.7}                    # tier-1 results move ratings more than tier-C/D ones
SOS_N = 12                                                             # opponents remembered for strength-of-schedule
FORM_N = 10
H2H_N = 10
TEST_FRACTION = 0.2
OUT = "model.json"
CACHE_DIR = "cache_lol"
LP_API = "https://lol.fandom.com/api.php"   # Leaguepedia, free and keyless: the backup source
LP_PAGE = 500                                # Cargo's per-request ceiling
LP_BACKOFF = [20, 45, 90, 180, 300]          # Fandom throttles hard; wait it out rather than give up
LP_SLEEP = 5.0                               # seconds between Leaguepedia calls; Fandom is strict
ACTIVE_DAYS = 150                            # a team counts as active if it played within this many days
API = "https://api.pandascore.co"
PNAMES = {}                     # player id -> nickname (filled by fetch_rosters)
TIER_NUM = {"s": 1, "a": 2, "b": 3, "c": 4, "d": 5}


# ------------------------------------------------------------------ data
class RateLimited(Exception):
    """PandaScore's hourly or monthly quota is spent. Raised instead of blocking for ten minutes
    so the run can finish on what it already has rather than dying in a CI job."""


def ps_get(path, params=None, tries=5, soft=False):
    if not TOKEN:
        sys.exit("No PandaScore token. Set the PANDASCORE_TOKEN environment variable,\n"
                 "or on GitHub add it under Settings -> Secrets and variables -> Actions.")
    for i in range(tries):
        try:
            r = requests.get(f"{API}{path}", params=params,
                             headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}, timeout=60)
        except requests.RequestException as e:
            if i == tries - 1: raise
            print(f"  network hiccup on {path}: {e}"); time.sleep(3 * (i + 1)); continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After") or 0) or 60 * (i + 1)
            if soft and i >= 1:
                raise RateLimited(path)
            print(f"  rate limited, waiting {wait}s"); time.sleep(wait); continue
        if r.status_code in (401, 403):
            sys.exit(f"PandaScore said {r.status_code}: check your token (or this endpoint needs a paid plan).")
        r.raise_for_status()
        return r.json()
    if soft: raise RateLimited(path)
    raise RuntimeError(f"gave up on {path}")


def fetch_past_matches(months):
    os.makedirs(CACHE_DIR, exist_ok=True)
    f = os.path.join(CACHE_DIR, "matches.json")
    have = {}
    if os.path.exists(f):
        for m in json.load(open(f, encoding="utf-8")):
            have[m["id"]] = m
        print(f"cache: {len(have)} matches")
    since = datetime.now(timezone.utc) - timedelta(days=30 * months)
    newest = max((m["begin_at"] for m in have.values()), default="")
    lo = max(since.isoformat(), (newest[:10] + "T00:00:00Z") if newest else "")
    hi = datetime.now(timezone.utc).isoformat()

    page, added, stopped = 1, 0, False
    while True:
        try:
            rows = ps_get("/lol/matches/past", {
                "sort": "-begin_at", "per_page": 100, "page": page,
                "range[begin_at]": f"{lo},{hi}", "filter[status]": "finished",
            }, soft=True)
        except RateLimited:
            print(f"  PandaScore quota is spent at page {page}. Keeping the {len(have)} matches already")
            print("  fetched and carrying on - run again in an hour and it will pick up where it left off.")
            stopped = True
            break
        if not rows:
            break
        for m in rows:
            if m.get("forfeit") or len(m.get("opponents") or []) != 2:
                continue
            if m["id"] not in have:
                added += 1
            have[m["id"]] = slim(m)
        print(f"  page {page}: {len(rows)} matches (total {len(have)})")
        if page % 5 == 0:                      # checkpoint, so a quota wall never costs you the whole run
            json.dump(sorted(have.values(), key=lambda m: (m["begin_at"], m["id"])),
                      open(f, "w", encoding="utf-8"))
        if len(rows) < 100:
            break
        page += 1
        time.sleep(0.7)   # free plan: 1000 req/hour
    matches = [m for m in have.values() if m["begin_at"] >= since.isoformat()]
    matches.sort(key=lambda m: (m["begin_at"], m["id"]))
    json.dump(matches, open(f, "w", encoding="utf-8"))
    print(f"downloaded {added} new, using {len(matches)} matches since {since:%Y-%m-%d}")
    return matches


def slim(m):
    """Keep only what we need from a PandaScore match."""
    opps = [o["opponent"] for o in m["opponents"]]
    maps = []
    for g in m.get("games") or []:
        w = g.get("winner") or {}
        if g.get("finished") and w.get("id") and not g.get("forfeit"):
            maps.append({"pos": g.get("position", 0), "winner": w["id"], "length": int(g.get("length") or 0)})
    maps.sort(key=lambda g: g["pos"])
    t = m.get("tournament") or {}
    return {
        "id": m["id"], "begin_at": m["begin_at"], "ts": int(datetime.fromisoformat(m["begin_at"].replace("Z", "+00:00")).timestamp()),
        "team_a": opps[0]["id"], "team_b": opps[1]["id"],
        "name_a": opps[0].get("name") or "", "name_b": opps[1].get("name") or "",
        "acr_a": opps[0].get("acronym") or "", "acr_b": opps[1].get("acronym") or "",
        "tournament_id": m.get("tournament_id") or t.get("id"),
        "tier": TIER_NUM.get((t.get("tier") or "").lower(), 4),
        "league": ((m.get("league") or {}).get("name") or ""),
        "bo": m.get("number_of_games") or len(maps) or 1,
        "games": maps, "winner": m.get("winner_id"),
    }


def fetch_upcoming(days=7, back=1):
    """Matches from `back` days ago to `days` days ahead (PandaScore /lol/matches), slimmed.
    Yesterday's are included so the app can show results and settle picks."""
    lo = (datetime.now(timezone.utc) - timedelta(days=back)).isoformat(timespec="seconds")
    hi = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat(timespec="seconds")
    out, page = [], 1
    while True:
        try:
            rows = ps_get("/lol/matches", {"sort": "begin_at", "per_page": 100, "page": page,
                                           "range[begin_at]": f"{lo},{hi}"}, soft=True) or []
        except RateLimited:
            print(f"  quota spent while reading the schedule - using the {len(out)} matches fetched so far")
            break
        for m in rows:
            opps = [o.get("opponent") or {} for o in (m.get("opponents") or [])]
            if len(opps) != 2 or not opps[0].get("id") or not opps[1].get("id") or not m.get("begin_at"): continue
            t = m.get("tournament") or {}; sr = m.get("serie") or {}; lg = m.get("league") or {}
            out.append({"id": m["id"], "ts": int(datetime.fromisoformat(m["begin_at"].replace("Z", "+00:00")).timestamp()), "team_a": opps[0]["id"], "team_b": opps[1]["id"],
                        "name_a": opps[0].get("name") or "", "name_b": opps[1].get("name") or "", "acr_a": opps[0].get("acronym") or "", "acr_b": opps[1].get("acronym") or "",
                        "tournament_id": m.get("tournament_id") or t.get("id"), "tier": TIER_NUM.get((t.get("tier") or "").lower(), 4), "bo": m.get("number_of_games") or 3,
                        "status": m.get("status") or "", "winner": (m.get("winner") or {}).get("id"),
                        "score": [ (r.get("score") if isinstance(r, dict) else None) for r in (m.get("results") or []) ][:2],
                        "event": " ".join(x for x in [lg.get("name") or "", sr.get("full_name") or sr.get("name") or "", t.get("name") or ""] if x).strip(), "name": m.get("name") or ""})
        if len(rows) < 100: break
        page += 1; time.sleep(0.7)
    print(f"schedule: {len(out)} matches from {back}d ago to {days}d ahead")
    return out


def fetch_rosters(tournament_ids):
    """tournament_id -> {team_id: [player_ids]} via GET /tournaments/{id}/rosters (free plan)."""
    f = os.path.join(CACHE_DIR, "rosters.json"); pf = os.path.join(CACHE_DIR, "players.json")
    have = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    if os.path.exists(pf): PNAMES.update({int(k): v for k, v in json.load(open(pf, encoding="utf-8")).items()})
    if have and not PNAMES: print("  roster cache has no player names yet - refetching once so the app can show names"); have = {}
    todo = [t for t in tournament_ids if t and str(t) not in have]
    print(f"rosters: {len(have)} cached, {len(todo)} to fetch")
    for i, tid in enumerate(todo):
        try:
            res = ps_get(f"/tournaments/{tid}/rosters")
            rosters = res.get("rosters") if isinstance(res, dict) else res
            have[str(tid)] = {str(r["id"]): [p["id"] for p in (r.get("players") or [])]
                              for r in (rosters or []) if r.get("id")}
            for r in (rosters or []):
                for p in (r.get("players") or []):
                    if p.get("id"): PNAMES[p["id"]] = p.get("name") or p.get("slug") or str(p["id"])
        except RateLimited:
            print(f"  quota spent after {i} of {len(todo)} rosters. Saving and carrying on with what we have;")
            print("  the rest will be picked up on the next run.")
            json.dump(have, open(f, "w", encoding="utf-8"))
            json.dump({str(k): v for k, v in PNAMES.items()}, open(pf, "w", encoding="utf-8"))
            break
        except SystemExit:
            raise
        except Exception as e:
            print(f"  roster {tid} failed: {e}")
            have[str(tid)] = {}
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(todo)}")
            json.dump(have, open(f, "w", encoding="utf-8")); json.dump({str(k): v for k, v in PNAMES.items()}, open(pf, "w", encoding="utf-8"))
        time.sleep(0.7)
    json.dump(have, open(f, "w", encoding="utf-8")); json.dump({str(k): v for k, v in PNAMES.items()}, open(pf, "w", encoding="utf-8"))
    return have


def fetch_player_names(ids):
    """Fill in nicknames for any player id we do not have a name for.

    The roster endpoint returns names, but a cache built by an older run can hold ids with no
    names, and the old "refetch if we have none" check never fires once a single name is present.
    /lol/players takes 100 ids at a time, so healing the whole cache costs about 20 requests."""
    todo = sorted({int(p) for p in ids if p and int(p) not in PNAMES})
    if not todo:
        return
    print(f"player names: {len(PNAMES)} known, looking up {len(todo)} missing")
    pf = os.path.join(CACHE_DIR, "players.json")
    got = 0
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        try:
            rows = ps_get("/lol/players", {"filter[id]": ",".join(map(str, chunk)), "per_page": 100}, soft=True)
        except RateLimited:
            print(f"  quota spent after {got} names - the rest stay as ids until the next run")
            break
        except Exception as e:
            print(f"  name lookup failed: {e}")
            break
        for p in rows or []:
            if p.get("id"):
                PNAMES[p["id"]] = p.get("name") or p.get("slug") or str(p["id"])
                got += 1
        os.makedirs(CACHE_DIR, exist_ok=True)
        json.dump({str(k): v for k, v in PNAMES.items()}, open(pf, "w", encoding="utf-8"))
        time.sleep(0.7)
    print(f"  resolved {got} names")


def norm_team(x):
    """loose team-name key so the same org from two sources lands on one id."""
    x = re.sub(r"\b(team|esports?|e-sports|gaming|club|academy|the)\b", " ", str(x or "").lower())
    return re.sub(r"[^a-z0-9]", "", x)


class LPRateLimited(Exception):
    """Leaguepedia asked us to slow down. It reports this as a JSON error body with HTTP 200,
    not as a 429, so it has to be read out of the payload."""


def lp_get(params, tries=5, quiet=False):
    """One Leaguepedia Cargo query. Keyless, but Fandom throttles hard and answers a throttle
    with a normal 200 whose body is an error. Back off and retry rather than treating it as fatal."""
    q = {"action": "cargoquery", "format": "json", "limit": LP_PAGE}
    q.update(params)
    for i in range(tries):
        try:
            r = requests.get(LP_API, params=q, timeout=60,
                             headers={"User-Agent": "LoLModelTrainer/2.0 (personal use; contact via GitHub)"})
            if r.status_code in (429, 503):
                wait = int(r.headers.get("Retry-After") or 0) or LP_BACKOFF[min(i, len(LP_BACKOFF) - 1)]
                if not quiet: print(f"    Leaguepedia throttled (HTTP {r.status_code}), waiting {wait}s")
                time.sleep(wait); continue
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                info = str(j["error"].get("info", "cargo error"))
                if "rate limit" in info.lower() or "exceeded" in info.lower():
                    wait = LP_BACKOFF[min(i, len(LP_BACKOFF) - 1)]
                    if not quiet: print(f"    Leaguepedia rate limit, waiting {wait}s")
                    time.sleep(wait); continue
                raise RuntimeError(info)            # a real schema error: do not retry, surface it
            return [row.get("title", {}) for row in j.get("cargoquery", [])]
        except requests.RequestException:
            if i == tries - 1: raise
            time.sleep(3 * (i + 1))
    raise LPRateLimited("Leaguepedia kept throttling")


def fetch_leaguepedia(months, name2id, want_players=True):
    """Backup source. PandaScore misses a lot of small regional play; Leaguepedia has it, plus the
    two things PandaScore will not tell you: the patch each game was played on, and who actually
    walked on stage (which is how stand-ins get caught).

    Returns (matches, patch_by_key, roles) where matches use the same shape as slim()."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cf = os.path.join(CACHE_DIR, "leaguepedia.json")
    cache = json.load(open(cf, encoding="utf-8")) if os.path.exists(cf) else {"games": {}, "players": {}}
    since = (datetime.now(timezone.utc) - timedelta(days=30 * months)).strftime("%Y-%m-%d %H:%M:%S")
    newest = max((g.get("dt", "") for g in cache["games"].values()), default="")
    lo = max(since, newest) if newest else since

    print(f"Leaguepedia: {len(cache['games'])} games cached, fetching from {lo[:10]}")
    off = 0; added = 0
    while True:
        try:
            rows = lp_get({"tables": "ScoreboardGames=SG",
                           "fields": ("SG.GameId=gid,SG.Tournament=tourn,SG.Team1=t1,SG.Team2=t2,SG.WinTeam=win,"
                                      "SG.Gamelength_Number=len,SG.DateTime_UTC=dt,SG.Patch=patch,SG.Team1Score=s1,SG.Team2Score=s2"),
                           "where": f"SG.DateTime_UTC >= '{lo}'",
                           "order_by": "SG.DateTime_UTC ASC", "offset": off})
        except Exception as e:
            print(f"  Leaguepedia stopped at offset {off}: {e}")
            break
        if not rows: break
        for r in rows:
            gid = r.get("gid")
            if not gid: continue
            if gid not in cache["games"]: added += 1
            cache["games"][gid] = r
        off += len(rows)
        if len(rows) < LP_PAGE: break
        if off % (LP_PAGE * 10) == 0:
            print(f"  {off} games"); json.dump(cache, open(cf, "w", encoding="utf-8"))
        time.sleep(LP_SLEEP)
    print(f"  {added} new games, {len(cache['games'])} total")

    if want_players:
        need = [g for g in cache["games"] if g not in cache["players"]]
        print(f"  players: {len(need)} games still need their scoreboard")
        done = 0
        for i in range(0, len(need), 40):                 # 40 games at a time keeps the WHERE clause sane
            chunk = need[i:i + 40]
            inlist = ",".join("'" + g.replace("'", "''") + "'" for g in chunk)
            try:
                rows = lp_get({"tables": "ScoreboardPlayers=SP",
                               "fields": "SP.GameId=gid,SP.Link=player,SP.Team=team,SP.Role=role,SP.Side=side",
                               "where": f"SP.GameId IN ({inlist})", "limit": LP_PAGE})
            except LPRateLimited:
                print(f"  Leaguepedia is still throttling after {done} of {len(need)} games.")
                print("  Keeping what is cached - run again later and it resumes from here.")
                break
            except Exception as e:
                print(f"  scoreboard fetch stopped: {e}")
                break
            by = defaultdict(list)
            for r in rows:
                if r.get("gid"): by[r["gid"]].append(r)
            for g in chunk:
                cache["players"][g] = by.get(g, [])
            done += len(chunk)
            if done % 400 == 0:
                print(f"    {done}/{len(need)}"); json.dump(cache, open(cf, "w", encoding="utf-8"))
            time.sleep(LP_SLEEP)
    json.dump(cache, open(cf, "w", encoding="utf-8"))

    # ---- fold game rows into matches, the same shape slim() produces
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30 * months)).isoformat()
    bymatch = {}
    patches = {}
    roles = {}
    for gid, g in cache["games"].items():
        dt = g.get("dt") or ""
        if not dt or not g.get("t1") or not g.get("t2"): continue
        try:
            ts = int(datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            continue
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if iso < cutoff: continue
        n1, n2 = norm_team(g["t1"]), norm_team(g["t2"])
        a = name2id.get(n1) or ("lp-" + n1)
        b = name2id.get(n2) or ("lp-" + n2)
        if a == b: continue
        key = (min(str(a), str(b)), max(str(a), str(b)), ts // 10800)   # same two teams inside three hours = one series
        M = bymatch.get(key)
        if not M:
            M = bymatch[key] = {"id": "lp" + str(gid), "begin_at": iso, "ts": ts, "team_a": a, "team_b": b,
                                "name_a": g["t1"], "name_b": g["t2"], "acr_a": "", "acr_b": "",
                                "tournament_id": None, "tier": 4, "league": g.get("tourn") or "",
                                "bo": 1, "games": [], "winner": None, "source": "leaguepedia"}
        wn = norm_team(g.get("win") or "")
        wid = a if wn == n1 else (b if wn == n2 else None)
        if wid is None: continue
        try: ln = int(float(g.get("len") or 0) * 60)
        except Exception: ln = 0
        M["games"].append({"pos": len(M["games"]) + 1, "winner": wid, "length": ln})
        if g.get("patch"): patches[str(gid)] = g["patch"]
        for pr in cache["players"].get(gid, []):
            pl = pr.get("player")
            if pl and pr.get("role"):
                roles[pl] = pr["role"].lower()
    out = []
    for M in bymatch.values():
        if not M["games"]: continue
        M["bo"] = len(M["games"])
        wa = sum(1 for g in M["games"] if g["winner"] == M["team_a"])
        M["winner"] = M["team_a"] if wa * 2 > len(M["games"]) else M["team_b"]
        out.append(M)
    out.sort(key=lambda m: m["ts"])
    print(f"  built {len(out)} matches from Leaguepedia, {len(patches)} with a patch, {len(roles)} players with a role")
    return out, patches, roles


def merge_sources(primary, extra):
    """Keep every PandaScore match; add a Leaguepedia one only when it is not the same series."""
    seen = set()
    for m in primary:
        seen.add((min(str(m["team_a"]), str(m["team_b"])), max(str(m["team_a"]), str(m["team_b"])), m["ts"] // 10800))
    added = 0
    for m in extra:
        k = (min(str(m["team_a"]), str(m["team_b"])), max(str(m["team_a"]), str(m["team_b"])), m["ts"] // 10800)
        if k in seen: continue
        seen.add(k); primary.append(m); added += 1
    primary.sort(key=lambda m: (m["ts"], str(m["id"])))
    print(f"merged: {added} matches that PandaScore did not have")
    return primary


def demo_matches(n=2500, teams=40):
    random.seed(2)
    strength = {t: random.gauss(0, 120) for t in range(1, teams + 1)}
    rosters = {t: [t * 100 + i for i in range(5)] for t in strength}
    now = int(time.time()); out = []
    for i in range(n):
        a, b = random.sample(list(strength), 2)
        p = 1 / (1 + 10 ** (-(strength[a] - strength[b]) / 400))
        bo = random.choice([1, 3, 3, 3, 5]); wins = [0, 0]; maps = []
        while max(wins) < (bo + 1) // 2:
            w = a if random.random() < p else b; lose = random.randint(3, 11) if random.random() < 0.85 else random.randint(12, 13)
            wins[0 if w == a else 1] += 1; maps.append({"pos": len(maps) + 1, "winner": w, "length": random.randint(1400, 3000)})
        ts = now - (n - i) * 3600 * 3
        out.append({"id": i, "begin_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), "ts": ts,
                    "team_a": a, "team_b": b, "name_a": f"Team {a}", "name_b": f"Team {b}", "acr_a": f"T{a}", "acr_b": f"T{b}",
                    "tournament_id": 1 + i // 200, "tier": random.choice([1, 2, 2, 3, 4]), "league": "Demo", "bo": bo,
                    "games": maps, "winner": a if wins[0] > wins[1] else b})
    ros = {str(t): {str(k): v for k, v in rosters.items()} for t in range(1, 1 + n // 200 + 1)}
    for t, pl in rosters.items():
        for k, p in enumerate(pl): PNAMES[p] = f"p{t}_{k+1}"
    PNAMES[99999] = "standin"
    up = []
    for i in range(12):
        a, b = random.sample(list(strength), 2); ts = now + 3600 * (6 + i * 5) - (20 * 3600 if i < 3 else 0)
        up.append({"status": "not_started", "winner": None, "score": [], "id": 900000 + i, "ts": ts, "team_a": a, "team_b": b, "name_a": f"Team {a}", "name_b": f"Team {b}", "acr_a": f"T{a}", "acr_b": f"T{b}", "tournament_id": 1 + n // 200,
                   "tier": random.choice([1, 2, 3]), "bo": random.choice([3, 3, 5]), "event": "Demo Masters - Playoffs", "name": f"Team {a} vs Team {b}"})
    if random.random() < 2: ros[str(1 + n // 200)][str(up[0]["team_a"])] = rosters[up[0]["team_a"]][:4] + [99999]     # a stand-in, to show the roster-change warning
    return out, ros, up


# ------------------------------------------------------------------ ratings & features
def h2h_key(a, b):
    """Team ids are ints from PandaScore and strings like "lp-t1" from Leaguepedia, so they cannot
    be compared with min()/max(). Order the pair by its string form instead, and key on strings so
    the app can rebuild the same key in JavaScript."""
    sa, sb = str(a), str(b)
    return (sa, sb) if sa <= sb else (sb, sa)


def expected(ra, rb): return 1 / (1 + 10 ** (-(ra - rb) / 400))


class State:
    def __init__(self):
        self.team_elo = defaultdict(lambda: BASE); self.player_elo = defaultdict(lambda: BASE)
        self.form = defaultdict(lambda: deque(maxlen=FORM_N)); self.last_ts = {}
        self.roster = {}; self.h2h = defaultdict(lambda: deque(maxlen=H2H_N))
        self.names = {}; self.acr = {}; self.games = defaultdict(int)
        self.lineups = defaultdict(dict); self.pgames = defaultdict(int); self.plast = {}; self.pteam = {}
        self.entry = {}; self.opp_elo = defaultdict(lambda: deque(maxlen=SOS_N)); self.tiers = defaultdict(lambda: deque(maxlen=SOS_N))
        self.patch_games = defaultdict(int)      # games this team has played on the current patch
        self.cur_patch = None
        self.roles = {}                          # player -> top/jungle/mid/bot/support, from Leaguepedia
        self.pool = defaultdict(lambda: defaultdict(lambda: {"games": 0, "last": 0}))   # team -> player -> usage
        self.n_games = 0

    def recentre(self):
        """Elo only means anything relative to the pool it came from. These pools take in new teams
        at a tier-based entry rating and lose old ones silently, so without a pull back toward BASE
        the whole scale drifts and a 1600 team this year is not a 1600 team next year."""
        act = [t for t in self.team_elo if self.games[t] >= 5]
        if len(act) >= 12:
            off = sum(self.team_elo[t] for t in act) / len(act) - BASE
            if abs(off) > 0.5:
                for t in list(self.team_elo): self.team_elo[t] -= off
        actp = [p for p in self.player_elo if self.pgames.get(p, 0) >= 5]
        if len(actp) >= 25:
            off = sum(self.player_elo[p] for p in actp) / len(actp) - BASE
            if abs(off) > 0.5:
                for p in list(self.player_elo): self.player_elo[p] -= off

    def set_patch(self, patch):
        """A new patch resets the adaptation counter: who has had reps on THIS version of the game."""
        if patch and patch != self.cur_patch:
            self.cur_patch = patch
            self.patch_games = defaultdict(int)

    def stand_ins(self, t, players):
        """players on the card who have barely played for this team."""
        if not players: return 0.0
        return float(sum(1 for p in players if self.pool[t][p]["games"] < 3))

    def role_elo(self, players):
        """mean player Elo, but only over the five distinct roles when roles are known, so a team
        that lists six names or a missing support does not quietly change the average."""
        if not players: return None
        byrole = {}
        for p in players:
            r = self.roles.get(p)
            if r and r not in byrole: byrole[r] = self.player_elo[p]
        if len(byrole) >= 4:
            return float(np.mean(list(byrole.values())))
        return None

    def enter(self, t, tier, ts):
        """first sighting: start at the tier's base rating. Long idle: regress a fifth of the way back to it."""
        if t not in self.entry:
            self.entry[t] = TIER_BASE.get(tier, 1400.0); self.team_elo[t] = self.entry[t]
        elif ts - self.last_ts.get(t, ts) > 60 * 86400:
            self.team_elo[t] += 0.2 * (self.entry[t] - self.team_elo[t])
    def enter_players(self, players, tier):
        for p in players or []:
            if p not in self.player_elo: self.player_elo[p] = TIER_BASE.get(tier, 1400.0)
    def sos(self, t): o = self.opp_elo[t]; return float(np.mean(o)) if o else self.team_elo[t]
    def tier_avg(self, t): o = self.tiers[t]; return float(np.mean(o)) if o else 3.0

    def team_feats(self, t, players, now):
        r = self.roster.get(t)
        stab = (len(set(players) & set(r)) / max(len(r), 1)) if (r and players) else 0.5
        f = self.form[t]; form = (sum(f) / len(f)) if f else 0.5
        pe = [self.player_elo[p] for p in players] if players else [self.team_elo[t]]
        rest = min((now - self.last_ts.get(t, now - 14 * 86400)) / 86400, 14)
        return self.team_elo[t], float(np.mean(pe)), form, stab, rest, self.games[t]

    def features(self, a, b, pa_, pb_, ts, tier):
        self.enter(a, tier, ts); self.enter(b, tier, ts); self.enter_players(pa_, tier); self.enter_players(pb_, tier)
        ea, pa, fa, sa, ra, ga = self.team_feats(a, pa_, ts)
        eb, pb, fb, sb, rb, gb = self.team_feats(b, pb_, ts)
        key = h2h_key(a, b); h = self.h2h[key]
        if h:
            wa = sum(h) if str(a) == key[0] else len(h) - sum(h); h2h = (wa - (len(h) - wa)) / len(h)
        else:
            h2h = 0.0
        rea = self.role_elo(pa_); reb = self.role_elo(pb_)
        role_diff = ((rea - reb) / 100) if (rea is not None and reb is not None) else ((pa - pb) / 100)
        return [(ea - eb) / 100, (pa - pb) / 100, fa - fb, sa - sb, h2h, (ra - rb) / 7,
                math.log1p(ga) - math.log1p(gb), 1.0 if tier == 1 else 0.0,
                (self.sos(a) - self.sos(b)) / 100, self.tier_avg(b) - self.tier_avg(a),
                (self.sos(a) - self.sos(b)) / 100 * ((ea - eb) / 100),
                role_diff,
                (math.log1p(self.patch_games[a]) - math.log1p(self.patch_games[b])),
                (self.stand_ins(b, pb_) - self.stand_ins(a, pa_)) / 5.0]

    def update_map(self, a, b, pa_, pb_, a_won, ts, tier=3, length=0):
        win = 1.0 if a_won else 0.0
        e = expected(self.team_elo[a], self.team_elo[b])
        mov = 1.0
        if length:                                                                  # a 24-minute stomp says more than a 45-minute grind
            mov = 1.35 if length < 1620 else (1.15 if length < 1920 else (1.0 if length < 2280 else (0.85 if length < 2700 else 0.75)))
        kt = TIER_K.get(tier, 1.0) * mov
        ka = K_TEAM * kt * (1.5 if self.games[a] < 10 else 1.0); kb = K_TEAM * kt * (1.5 if self.games[b] < 10 else 1.0)
        ea0, eb0 = self.team_elo[a], self.team_elo[b]
        self.team_elo[a] += ka * (win - e); self.team_elo[b] += kb * ((1 - win) - (1 - e))
        self.opp_elo[a].append(eb0); self.opp_elo[b].append(ea0); self.tiers[a].append(tier); self.tiers[b].append(tier)
        if pa_ and pb_:
            ep = expected(np.mean([self.player_elo[p] for p in pa_]), np.mean([self.player_elo[p] for p in pb_]))
            for p in pa_: self.player_elo[p] += K_PLAYER * kt * (win - ep)
            for p in pb_: self.player_elo[p] += K_PLAYER * kt * ((1 - win) - (1 - ep))
        for t, pl in ((a, pa_), (b, pb_)):
            if pl:
                key = ",".join(str(x) for x in sorted(pl)); L = self.lineups[t].get(key) or {"games": 0, "first": ts, "last": ts}
                L["games"] += 1; L["last"] = ts; self.lineups[t][key] = L
                for x in pl: self.pgames[x] += 1; self.plast[x] = ts; self.pteam[x] = t
        self.patch_games[a] += 1; self.patch_games[b] += 1
        for t, pl in ((a, pa_), (b, pb_)):
            for x in (pl or []):
                self.pool[t][x]["games"] += 1; self.pool[t][x]["last"] = ts
        self.n_games += 1
        if self.n_games % 1500 == 0: self.recentre()
        self.form[a].append(win); self.form[b].append(1 - win)
        self.last_ts[a] = self.last_ts[b] = ts
        key = h2h_key(a, b); self.h2h[key].append(win if str(a) == key[0] else 1 - win)
        self.games[a] += 1; self.games[b] += 1

    def after_match(self, m, pa_, pb_):
        if pa_: self.roster[m["team_a"]] = list(pa_)
        if pb_: self.roster[m["team_b"]] = list(pb_)
        for k, nm, ac in (("team_a", "name_a", "acr_a"), ("team_b", "name_b", "acr_b")):
            if m[nm]: self.names[m[k]] = m[nm]
            if m[ac]: self.acr[m[k]] = m[ac]


FEATURE_NAMES = ["team_elo_diff", "player_elo_diff", "form_diff", "roster_stability_diff",
                 "head_to_head", "rest_diff", "experience_diff", "tier1", "schedule_strength_diff",
                 "tier_played_diff", "elo_x_schedule", "role_elo_diff", "patch_reps_diff", "stand_in_diff"]
NM = len(FEATURE_NAMES)          # the base model never sees the market; it is blended in afterwards (stack)
# ------------------------------------------------------------------ market feature (from odds_logger.py)
def _mnorm(s):
    import re as _re
    s = _re.sub(r"\b(team|esports?|e-sports|gaming|club|the)\b", " ", str(s or "").lower())
    return _re.sub(r"[^a-z0-9]", "", s)

def _same(a, b):
    a, b = _mnorm(a), _mnorm(b)
    if not a or not b: return False
    return a == b or a in b or b in a or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4])

def load_market(path, game):
    """Returns list of (ts, home, away, p_home_vigfree_opening) for one game from odds_log.json."""
    import json as _json, os as _os
    from datetime import datetime as _dt, timezone as _tz
    if not path or not _os.path.exists(path): return []
    out = []
    for rec in _json.load(open(path, encoding="utf-8")).values():
        if rec.get("game") != game or not rec.get("first", {}).get("ml"): continue
        books = list(rec["first"]["ml"].values())
        h, a = books[0]
        if h <= 1 or a <= 1: continue
        qh, qa = 1 / h, 1 / a; p = qh / (qh + qa)
        try: ts = int(_dt.fromisoformat(rec["date"].replace("Z", "+00:00")).timestamp())
        except Exception: continue
        out.append((ts, rec.get("home", ""), rec.get("away", ""), p))
    print(f"market: {len(out)} logged {game} events with opening moneylines")
    return out

def market_feats(market, ts, name_a, name_b, window_h=8):
    """[logit of opening P(team A), has_market]. Picks the CLOSEST logged game in time (teams often meet
    on consecutive days, so 'first match within 36h' picked the wrong game). Window default 8 hours."""
    import math as _m
    best, best_dt = None, None
    for (mts, h, a, p) in market:
        dt = abs(mts - ts)
        if dt > window_h * 3600: continue
        if _same(h, name_a) and _same(a, name_b): cand = p
        elif _same(h, name_b) and _same(a, name_a): cand = 1 - p
        else: continue
        if best_dt is None or dt < best_dt: best, best_dt = cand, dt
    if best is None: return [0.0, 0.0]
    best = min(max(best, 0.02), 0.98)
    return [_m.log(best / (1 - best)), 1.0]



# ------------------------------------------------------------------ model
def choose_l2(X, y, grid=(0.3, 1, 3, 10, 30, 100, 300), blocks=5):
    """Pick the ridge strength by out-of-fold log loss, always training on the past. We bet the
    probability, not the argmax, so log loss is what to minimise."""
    m = len(X); best = (1.0, 1e9)
    for l2 in grid:
        tot = 0.0; cnt = 0
        for bi in range(1, blocks):
            cut = m * bi // blocks; hi = m * (bi + 1) // blocks
            mod = train_logistic(X[:cut], y[:cut], l2=l2)
            p = np.clip(predict(mod, X[cut:hi]), 1e-9, 1 - 1e-9); yy = y[cut:hi]
            tot += float(-np.sum(yy * np.log(p) + (1 - yy) * np.log(1 - p))); cnt += len(yy)
        ll = tot / max(cnt, 1)
        print(f"    l2={l2:<6} out-of-fold log loss {ll:.4f}")
        if ll < best[1]: best = (l2, ll)
    print(f"  chose l2={best[0]}")
    return best[0]


def fit_temperature(X, y, l2, blocks=5):
    """p' = sigmoid(t * logit(p)), t fitted only on games the model never trained on. A model that
    says 85% and wins 76% is wrong exactly where Kelly stakes the most."""
    m = len(X); L = []; Y = []
    for bi in range(1, blocks):
        cut = m * bi // blocks; hi = m * (bi + 1) // blocks
        mod = train_logistic(X[:cut], y[:cut], l2=l2)
        p = np.clip(predict(mod, X[cut:hi]), 1e-6, 1 - 1e-6)
        L.append(np.log(p / (1 - p))); Y.append(y[cut:hi])
    if not L: return 1.0
    L = np.concatenate(L); Y = np.concatenate(Y)
    best = (1.0, 1e9)
    for t in np.arange(0.40, 1.41, 0.02):
        q = np.clip(1 / (1 + np.exp(-t * L)), 1e-9, 1 - 1e-9)
        ll = float(-np.mean(Y * np.log(q) + (1 - Y) * np.log(1 - q)))
        if ll < best[1]: best = (float(t), ll)
    print(f"  calibration temperature {best[0]:.2f} (1.00 = already calibrated, lower = overconfident)")
    return best[0]


def train_logistic(X, y, l2=1.0, iters=40, lr=None):
    """L2-regularised logistic regression solved by Newton's method (converges exactly, unlike gradient descent)."""
    mu = X.mean(0); sd = X.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd)      # a constant feature must not be divided into oblivion
    Xs = (X - mu) / sd; n, d = Xs.shape
    Xb = np.hstack([Xs, np.ones((n, 1))]); w = np.zeros(d + 1); R = np.eye(d + 1) * l2; R[d, d] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Xb @ w))); g = Xb.T @ (p - y) + R @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8: break
    return {"w": w[:d], "b": float(w[d]), "mu": mu, "sd": sd}

def predict(model, X): return 1 / (1 + np.exp(-(((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"])))
def apply_temp(p, t): p = np.clip(p, 1e-9, 1 - 1e-9); return 1 / (1 + np.exp(-t * np.log(p / (1 - p))))

def evaluate(name, p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); acc = ((p > 0.5) == (y == 1)).mean()
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(); brier = ((p - y) ** 2).mean()
    print(f"  {name:<22} accuracy {acc*100:5.1f}%   log-loss {ll:.4f}   brier {brier:.4f}"); return acc, ll, brier

def calibration(p, y, bins=8):
    edges = np.linspace(0, 1, bins + 1); rows = []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if mask.sum() >= 10:
            rows.append({"bin": f"{edges[i]:.2f}-{edges[i+1]:.2f}", "n": int(mask.sum()),
                         "predicted": round(float(p[mask].mean()), 3), "actual": round(float(y[mask].mean()), 3)})
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--lp", action="store_true",
                    help="also pull Leaguepedia (small leagues, patches, roles). Fandom throttles "
                         "anonymous API use hard, so this is slow and often only gets partway - but it "
                         "caches, so running it now and then keeps topping the cache up.")
    ap.add_argument("--lp-players", action="store_true",
                    help="with --lp, also fetch per-game scoreboards for player roles. Slowest part by far.")
    ap.add_argument("--no-lp", action="store_true", help=argparse.SUPPRESS)          # old name, ignored
    ap.add_argument("--no-lp-players", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--check-sources", action="store_true",
                    help="fetch one small page from each source, print what came back, and stop")
    ap.add_argument("--out", default=OUT); ap.add_argument("--odds", default="odds_log.json"); args = ap.parse_args()

    if args.check_sources:
        print("PandaScore /lol/matches/past ...")
        if not TOKEN:
            print("  SKIPPED: PANDASCORE_TOKEN is not set in this shell.")
            print("           Leaguepedia needs no token, so the checks below still mean something.")
        else:
            try:
                r = ps_get("/lol/matches/past", {"per_page": 2, "sort": "-begin_at"})
                print(f"  ok, {len(r)} rows; first: {(r[0].get('name') if r else '-')}")
            except SystemExit as e:
                print(f"  FAILED: {e}")
            except Exception as e:
                print(f"  FAILED: {e}")
        print("Leaguepedia ScoreboardGames ...")
        try:
            rows = lp_get({"tables": "ScoreboardGames=SG",
                           "fields": "SG.GameId=gid,SG.Team1=t1,SG.Team2=t2,SG.WinTeam=win,SG.DateTime_UTC=dt,SG.Patch=patch",
                           "order_by": "SG.DateTime_UTC DESC", "limit": 3})
            print(f"  ok, {len(rows)} rows; first: {rows[0] if rows else '-'}")
        except Exception as e:
            print(f"  FAILED: {e}")
        print("Leaguepedia ScoreboardPlayers ...")
        try:
            rows = lp_get({"tables": "ScoreboardPlayers=SP",
                           "fields": "SP.GameId=gid,SP.Link=player,SP.Team=team,SP.Role=role",
                           "order_by": "SP.DateTime_UTC DESC", "limit": 3})
            print(f"  ok, {len(rows)} rows; first: {rows[0] if rows else '-'}")
        except Exception as e:
            print(f"  FAILED: {e}")
        return
    market = [] if args.demo else load_market(args.odds, "lol")
    moves = []                                                   # live price movement from odds_log.json, for the app
    if not args.demo and os.path.exists(args.odds):
        try:
            raw = json.load(open(args.odds, encoding="utf-8"))
            for k, r in raw.items():
                if r.get("game") != "lol" or r.get("closing") or not r.get("series"): continue
                ser = r["series"]
                moves.append({"a": r.get("home", ""), "b": r.get("away", ""), "date": r.get("date", ""),
                              "open": r.get("open_p", ser[0][1]), "now": r.get("last_p", ser[-1][1]),
                              "hi": r.get("hi_p"), "lo": r.get("lo_p"), "n": len(ser),
                              "spark": [x[1] for x in ser[-40:]], "vol": r.get("volume")})
        except Exception as e:
            print(f"  could not read price history from {args.odds}: {e}")
        print(f"price history: {len(moves)} open markets with a movement series")

    patches, lp_roles = {}, {}
    if args.demo:
        matches, rosters, upcoming = demo_matches()
    else:
        matches = fetch_past_matches(args.months)
        upcoming = fetch_upcoming(7, 1)
        if not args.lp:
            print("Leaguepedia: skipped (pass --lp to include it; it adds small leagues, patches and roles\n"
                  "             but Fandom throttles hard, so expect it to take a while and resume across runs)")
        rosters = fetch_rosters(sorted({m["tournament_id"] for m in matches + upcoming if m["tournament_id"]}))
        patches, lp_roles = {}, {}
        if args.lp:
            name2id = {}
            for m in matches + upcoming:
                for nm, tid in ((m.get("name_a"), m["team_a"]), (m.get("name_b"), m["team_b"])):
                    k = norm_team(nm)
                    if k: name2id.setdefault(k, tid)
            try:
                lp, patches, lp_roles = fetch_leaguepedia(args.months, name2id,
                                                          want_players=args.lp_players)
                matches = merge_sources(matches, lp)
            except Exception as e:
                print(f"Leaguepedia unavailable ({e}) - carrying on with PandaScore only")
    if len(matches) < 300:
        sys.exit(f"only {len(matches)} matches - not enough. Try --months 18 or check the token.")

    if not args.demo:
        seen_players = {p for ros in rosters.values() for pl in ros.values() for p in pl}
        try:
            fetch_player_names(seen_players)
        except Exception as e:
            print(f"  could not refresh player names: {e}")

    st = State(); st.roles = lp_roles or {}
    X, y, ts, MK = [], [], [], []
    no_roster = 0
    for m in matches:
        ros = rosters.get(str(m["tournament_id"]), {})
        pa = ros.get(str(m["team_a"])) or []; pb = ros.get(str(m["team_b"])) or []
        if not pa or not pb: no_roster += 1
        # random orientation so the intercept doesn't learn "first listed team"
        flip = random.Random(m["id"]).random() < 0.5
        a, b, PA, PB = (m["team_b"], m["team_a"], pb, pa) if flip else (m["team_a"], m["team_b"], pa, pb)
        mk = market_feats(market, m["ts"], m["name_b"] if flip else m["name_a"], m["name_a"] if flip else m["name_b"], 24)
        st.set_patch(m.get("patch"))
        for g in m["games"]:
            X.append(st.features(a, b, PA, PB, m["ts"], m["tier"])); MK.append(mk); y.append(1.0 if g["winner"] == a else 0.0); ts.append(m["ts"])
            st.update_map(a, b, PA, PB, g["winner"] == a, m["ts"], m["tier"], g.get("length"))
        st.after_match(m, pa, pb)
    X, y, MK = np.array(X), np.array(y), np.array(MK)
    nr = sum(1 for m in matches for g in m["games"] if g.get("length"))
    print(f"\n{len(matches)} matches, {len(X)} games ({nr} with a length for margin-of-victory Elo), {no_roster} matches without roster data")

    cut = int(len(X) * (1 - TEST_FRACTION)); Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
    print(f"train {len(Xtr)} games, test {len(Xte)} (from {datetime.fromtimestamp(ts[cut]):%Y-%m-%d})\n\nholdout results:")
    evaluate("coin flip", np.full(len(yte), ytr.mean()), yte)
    evaluate("team elo only", 1 / (1 + 10 ** (-(Xte[:, 0] * 100) / 400)), yte)
    evaluate("player elo only", 1 / (1 + 10 ** (-(Xte[:, 1] * 100) / 400)), yte)
    print("\nchoosing regularisation on the training games only:")
    L2 = choose_l2(Xtr, ytr)
    TEMP = fit_temperature(Xtr, ytr, L2)
    model = train_logistic(Xtr, ytr, l2=L2)
    pte_raw = predict(model, Xte); pte = apply_temp(pte_raw, TEMP)
    evaluate("logistic, raw", pte_raw, yte)
    acc, ll, brier = evaluate("logistic, calibrated", pte, yte)
    MKtr, MKte = MK[:cut], MK[cut:]; hm = MKte[:, 1] > 0; stack = None
    if hm.sum() >= 30:
        pm = 1 / (1 + np.exp(-MKte[hm, 0]))
        evaluate("market line alone", pm, yte[hm]); evaluate("base model, same games", pte[hm], yte[hm])
        ok = MKtr[:, 1] > 0
        if ok.sum() >= 100:
            oo = np.zeros(len(ytr)); edges = np.linspace(0, len(ytr), 6).astype(int)     # out-of-fold base predictions, chronological blocks
            for i in range(5):
                te_ = np.zeros(len(ytr), bool); te_[edges[i]:edges[i + 1]] = True
                oo[te_] = predict(train_logistic(Xtr[~te_], ytr[~te_]), Xtr[te_])
            lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
            bl = train_logistic(np.c_[lg(oo[ok]), MKtr[ok, 0]], ytr[ok], l2=0.1)
            ps = predict(bl, np.c_[lg(pte[hm]), MKte[hm, 0]]); evaluate("STACKED model+market", ps, yte[hm])
            wz = bl["w"] / bl["sd"]; stack = {"a": round(float(wz[0]), 5), "b": round(float(wz[1]), 5), "c": round(float(bl["b"] - (bl["mu"] / bl["sd"] * bl["w"]).sum()), 5)}
            print(f"  stacked blend: logit(p) = {wz[0]:+.2f}*model + {wz[1]:+.2f}*market (+const), fitted on {int(ok.sum())} train games with a logged line")
        print(f"  ({int(hm.sum())} holdout games had a logged opening line)")
    else:
        print(f"  only {int(hm.sum())} holdout games had a logged line - no stack yet, keep odds_logger running")
    print("\nfeature weights:")
    for n, w in sorted(zip(FEATURE_NAMES, model["w"]), key=lambda t: -abs(t[1])): print(f"  {n:<24} {w:+.3f}")
    cal = calibration(pte, yte); print("\ncalibration:")
    for r in cal: print(f"  {r['bin']}  n={r['n']:<4} predicted {r['predicted']:.2f}  actual {r['actual']:.2f}")

    final = train_logistic(X, y, l2=L2)
    active_since = time.time() - 120 * 86400
    teams = {str(t): {"name": st.names.get(t, f"team {t}"), "acronym": st.acr.get(t, ""), "elo": round(st.team_elo[t], 1), "sos": round(st.sos(t), 1), "tier_avg": round(st.tier_avg(t), 2), "entry": st.entry.get(t, BASE),
                      "form": [int(v) for v in st.form[t]], "last_ts": st.last_ts.get(t, 0),
                      "roster": st.roster.get(t, []), "games": st.games[t],
                      "patch_games": st.patch_games.get(t, 0),
                      "pool": [{"id": p, "games": v["games"], "last": v["last"]}
                               for p, v in sorted(st.pool[t].items(), key=lambda kv: (-kv[1]["last"], -kv[1]["games"]))[:18]]}
             for t in st.team_elo if st.last_ts.get(t, 0) >= active_since}
    # every pool player must survive the filter below, or the lineup dropdowns come up empty
    used = {p for t in teams.values() for p in t["roster"]} | {p["id"] for t in teams.values() for p in t["pool"]}
    players = {str(p): round(st.player_elo[p], 1) for p in st.player_elo if p in used}
    h2h = {f"{a}|{b}": [int(v) for v in d] for (a, b), d in st.h2h.items() if a in teams and b in teams}
    # player ranks among everyone who played in the last 120 days (1 = best)
    ranked = sorted([p for p in st.player_elo if st.plast.get(p, 0) >= active_since], key=lambda p: -st.player_elo[p]); rank = {p: i + 1 for i, p in enumerate(ranked)}
    def pinfo(p): return {"id": p, "name": PNAMES.get(p, f"player {p}"), "elo": round(st.player_elo[p], 1), "rank": rank.get(p), "of": len(ranked), "games": st.pgames.get(p, 0), "last_ts": st.plast.get(p, 0), "team": st.pteam.get(p)}
    def roster_report(t, expected, now):
        last = st.roster.get(t) or []; exp = list(expected or last)
        key = ",".join(str(x) for x in sorted(exp)); L = st.lineups[t].get(key) or {"games": 0, "first": None, "last": None}
        ins = [p for p in exp if p not in last]; outs = [p for p in last if p not in exp]
        best = max((v["games"] for v in st.lineups[t].values()), default=0)
        return {"expected": [pinfo(p) for p in exp], "in": [pinfo(p) for p in ins], "out": [pinfo(p) for p in outs], "known": bool(expected),
                "games_together": L["games"], "days_together": round((now - L["first"]) / 86400, 1) if L["first"] else 0, "best_lineup_games": best,
                "new_to_team": [pinfo(p) for p in exp if st.pteam.get(p) not in (None, t)], "stand_in": any(st.pgames.get(p, 0) < 5 for p in exp)}
    def _tid(t):
        """team keys are exported as strings; recover the original int id when there was one."""
        try: return int(t)
        except (TypeError, ValueError): return t

    now = int(time.time())
    for t in list(teams):                                    # a roster report for every team, so "Any two" has names and cohesion
        try:
            teams[t]["report"] = roster_report(_tid(t), None, now)
            for pl in teams[t]["report"]["expected"]: players[str(pl["id"])] = round(st.player_elo[pl["id"]], 1)
        except Exception:
            pass
    up_out = []
    for m in upcoming:
        for t, nm, ac in ((m["team_a"], m["name_a"], m["acr_a"]), (m["team_b"], m["name_b"], m["acr_b"])):
            st.names.setdefault(t, nm); st.acr.setdefault(t, ac)
        ros = rosters.get(str(m["tournament_id"]), {}); ra = ros.get(str(m["team_a"])) or []; rb = ros.get(str(m["team_b"])) or []
        for t in (m["team_a"], m["team_b"]):
            if str(t) not in teams: teams[str(t)] = {"name": st.names.get(t, f"team {t}"), "acronym": st.acr.get(t, ""), "elo": round(st.team_elo[t], 1), "sos": round(st.sos(t), 1), "tier_avg": round(st.tier_avg(t), 2),
                                                        "entry": st.entry.get(t, BASE), "form": [int(v) for v in st.form[t]], "last_ts": st.last_ts.get(t, 0), "roster": st.roster.get(t, []), "games": st.games[t],
                                                        "patch_games": st.patch_games.get(t, 0),
                                                        "pool": [{"id": p, "games": v["games"], "last": v["last"]}
                                                                 for p, v in sorted(st.pool[t].items(), key=lambda kv: (-kv[1]["last"], -kv[1]["games"]))[:18]]}
        up_out.append({"id": m["id"], "ts": m["ts"], "a": str(m["team_a"]), "b": str(m["team_b"]), "bo": m["bo"], "tier": m["tier"], "event": m["event"], "name": m["name"],
                       "status": m.get("status", ""), "winner": str(m["winner"]) if m.get("winner") else None, "score": m.get("score") or [],
                       "roster_a": roster_report(m["team_a"], ra, now), "roster_b": roster_report(m["team_b"], rb, now),
                       # the feature vector the trainer itself would build for this match. The app
                       # rebuilds the same vector in JavaScript so you can edit a lineup and get a
                       # real number; this field is what proves the two have not drifted apart.
                       "fv": [round(float(v), 6) for v in st.features(m["team_a"], m["team_b"],
                                                                     ra or st.roster.get(m["team_a"], []),
                                                                     rb or st.roster.get(m["team_b"], []),
                                                                     now, m["tier"])]})
    for u in up_out:
        for r in (u["roster_a"], u["roster_b"]):
            for p in r["expected"]: players[str(p["id"])] = round(st.player_elo[p["id"]], 1)
    recent = [{"id": m["id"], "ts": m["ts"], "a": str(m["team_a"]), "b": str(m["team_b"]), "winner": str(m["winner"]) if m.get("winner") else None,
               "score": [sum(1 for g in m["games"] if g["winner"] == m["team_a"]), sum(1 for g in m["games"] if g["winner"] == m["team_b"])]} for m in matches if m["ts"] >= now - 21 * 86400]
    out = {"game": "lol", "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "matches_used": len(matches), "games_used": int(len(X)),
           "base": BASE, "form_n": FORM_N, "features": FEATURE_NAMES, "stack": stack, "version": 3,
           "temp": round(float(TEMP), 4), "l2": float(L2), "cap": 0.85,
           "patch": st.cur_patch, "roles": {str(p): r for p, r in (st.roles or {}).items()},
           "sources": sorted({m.get("source", "pandascore") for m in matches}),
           "model": {"w": [round(float(v), 6) for v in final["w"]], "b": round(float(final["b"]), 6),
                     "mu": [round(float(v), 6) for v in final["mu"]], "sd": [round(max(float(v), 1e-6), 6) for v in final["sd"]]},
           "holdout": {"n": int(len(yte)), "accuracy": round(float(acc), 4), "logloss": round(float(ll), 4), "brier": round(float(brier), 4), "calibration": cal},
           "teams": teams, "players": players, "player_ranks": {str(p): rank[p] for p in ranked}, "players_ranked": len(ranked), "player_names": {str(p): PNAMES[p] for p in used if PNAMES.get(p)}, "h2h": h2h, "upcoming": up_out, "recent": recent, "moves": moves}
    json.dump(out, open(args.out, "w", encoding="utf-8"), separators=(",", ":"))
    print(f"\nwrote {args.out}: {len(teams)} active teams, {len(players)} players, {len(up_out)} upcoming matches, {os.path.getsize(args.out)//1024} KB")


if __name__ == "__main__":
    main()
