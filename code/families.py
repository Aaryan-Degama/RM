"""Title-template families with typed slots and umbrella pages; TempEL's slice
types; as-of-t titles.

tau(title) -> (template, slots). A slot is a title word that is a year, a
season ("2019–20"), an ordinal ("23rd"), a plain number, a strict Roman numeral
of >= 2 letters ("LIV"), or a number word ("second"). Words that mix letters and
digits ("COVID-19", "F-16") are never slots. Pages sharing a template are
editions of one family; the umbrella page is the template without its slot
("2020 FIFA World Cup" -> "FIFA World Cup").

  python code/families.py            # self-check
  python code/families.py coverage <test.jsonl or OUT.meta.jsonl>   # as-of-t sibling coverage
"""
import collections
import functools
import json
import re
import sys

PAGE_INFO = 'data/tempel/stats/snapshot_stats/page_info_{year}-01-01T00:00:00Z.csv'

WORD = re.compile(r'[^\W_]+(?:[–-][^\W_]+)*')
SEASON = re.compile(r'((?:1[5-9]|20)[0-9][0-9])[–-]([0-9][0-9]|(?:1[5-9]|20)[0-9][0-9])')
YEAR = re.compile(r'(?:1[5-9]|20)[0-9][0-9]')
ORDINAL = re.compile(r'([0-9]+)(?:st|nd|rd|th)', re.I)
ROMAN = re.compile(r'M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})')
NUMWORDS = ('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth').split()


def slot_of(word):
    """(kind, value) if the word is a slot, else None."""
    if m := SEASON.fullmatch(word):
        start, end = int(m.group(1)), m.group(2)
        end = int(end) if len(end) == 4 else start - start % 100 + int(end) + (100 if int(end) < start % 100 else 0)
        return ('season', (start, end))
    if YEAR.fullmatch(word):
        return ('year', int(word))
    if m := ORDINAL.fullmatch(word):
        return ('ordinal', int(m.group(1)))
    if word.isascii() and word.isdigit():  # not '¹' etc.
        return ('num', int(word))
    if len(word) >= 2 and word.isupper() and ROMAN.fullmatch(word):
        return ('roman', word.lower())
    if word.lower() in NUMWORDS:
        return ('numword', word.lower())
    return None


def norm(s):
    return ' '.join(s.replace('()', ' ').split())


@functools.lru_cache(maxsize=None)
def tau(title):
    """(template, slots); slots is a tuple of (kind, value). No slots -> no family of its own."""
    s = title.replace('_', ' ')
    out, slots, last = [], [], 0
    for m in WORD.finditer(s):
        sl = slot_of(m.group())
        if sl:
            out.append(s[last:m.start()] + '#')
            slots.append(sl)
            last = m.end()
    out.append(s[last:])
    return norm(''.join(out)).lower(), tuple(slots)


def umbrella_key(template):
    return norm(template.replace('#', ' '))


def families(titles):
    """Families inside one candidate list: list of sorted position lists (|F| >= 2, at least
    one slotted member). Groups by (template, slot kinds), adds umbrella pages, and merges
    groups that share a member (union-find), so the result is a partition of its members."""
    parent = list(range(len(titles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    groups = collections.defaultdict(list)
    plain = collections.defaultdict(list)
    for p, t in enumerate(titles):
        tmpl, slots = tau(t)
        if slots:
            groups[(tmpl, tuple(sorted({k for k, _ in slots})))].append(p)
        else:
            plain[norm(t.replace('_', ' ')).lower()].append(p)
    slotted = set()
    for (tmpl, _), members in groups.items():
        members = members + plain.get(umbrella_key(tmpl), [])
        slotted.update(p for p in members if tau(titles[p])[1])
        for p in members[1:]:
            parent[find(p)] = find(members[0])
    comp = collections.defaultdict(list)
    for p in range(len(titles)):
        comp[find(p)].append(p)
    fams = [sorted(c) for c in comp.values() if len(c) >= 2 and slotted & set(c)]
    assert len({p for f in fams for p in f}) == sum(map(len, fams))  # partition
    return fams


def filter_type(curr_test_year, target_orig_title):
    """Verbatim TempEL get_target_filter_type (s07_analyze_results_specific_year_v2.py:413),
    config look_in_entity_text=false; TempEL passes the title lowercased (line 504)."""
    target_orig_title = target_orig_title.lower()
    type = 'filter-other'
    curr_test_year = int(curr_test_year)
    year_str = str(curr_test_year - 1)
    if target_orig_title.startswith('20') or (year_str in target_orig_title) or \
            (str(curr_test_year) in target_orig_title) or (str(curr_test_year + 1) in target_orig_title):
        curr_test_year_range = '{}–{}'.format(year_str[-2:], str(curr_test_year)[-2:])
        should_be_in = ['league', 'liga', 'cup', 'election', 'tournament', 'liiga', 'tour',
                        'open', 'championship', 'prix', 'copa', 'season', 'olympics',
                        'team', 'football', curr_test_year_range, '_games', 'uefa', 'euro_', 'qualifying',
                        'campeonato', 'finals', '_nba_', 'basketball', '_final', '_olympic',
                        'segunda_di']
        for curr_sh_in in should_be_in:
            if curr_sh_in in target_orig_title.lower():
                type = 'filter-recurrent'
    for curr_keyword in ['covid', 'covid-19', 'coronavirus']:
        if curr_keyword in target_orig_title:
            type = 'filter-covid'
    return type


def page_titles(qids=None, year=2021):
    """numeric QID -> as-of-t Wikipedia title (first row wins). Streams the TSV;
    keeps only `qids` if given."""
    out = {}
    with open(PAGE_INFO.format(year=year), encoding='utf-8') as f:
        assert next(f).split('\t')[:3] == ['page_id', 'wikipedia_title', 'wikidata_qid']
        for line in f:
            _, title, qid = line.split('\t', 3)[:3]
            if not qid.startswith('Q'):
                continue
            q = int(qid[1:])
            if (qids is None or q in qids) and q not in out:
                out[q] = title
    return out


def coverage(path, year=2021):
    """For each new entity in a mention file: how many as-of-t pages share its template."""
    meta = [json.loads(l) for l in open(path, encoding='utf-8')]
    new = {m['target_qid']: m['target_title'] for m in meta if m['category'] == 'new_entities'}
    count = collections.Counter(tau(t)[0] for t in page_titles(year=year).values() if tau(t)[1])
    sib = {q: count[tau(t)[0]] - 1 for q, t in new.items() if tau(t)[1]}
    has = sorted(n for n in sib.values() if n >= 1)
    print(f'new entities {len(new)}; with a slot {len(sib)}; with >=1 as-of-t sibling {len(has)}; '
          f'median siblings {has[len(has) // 2] if has else 0}')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'coverage':
        coverage(sys.argv[2])
    else:
        assert tau('2022_FIFA_World_Cup') == ('# fifa world cup', (('year', 2022),))
        assert tau('2020–21_Serie_B') == ('# serie b', (('season', (2020, 2021)),))
        assert tau('1999–00_NHL_season')[1] == (('season', (1999, 2000)),)
        assert tau('2008–09_Serie_B')[0] == tau('2020–21_Serie_B')[0]
        assert tau('23rd_Tour_of_X')[1] == (('ordinal', 23),)
        assert tau('Super_Bowl_LIV')[1] == (('roman', 'liv'),)
        assert tau('The_Masked_Singer_(American_season_4)')[1] == (('num', 4),)
        assert slot_of('¹') is None and slot_of('٢٠٢٠') is None
        for t in ('Barack_Obama', 'COVID-19_pandemic_in_Spain', 'F-16_Fighting_Falcon',
                  'American_Civil_War', 'CIVIL_rights', 'I_Love_Lucy', 'Malcolm_X'):
            assert tau(t)[1] == (), (t, tau(t))
        assert umbrella_key(tau('2020_FIFA_World_Cup')[0]) == 'fifa world cup'
        # editions + umbrella form one family; the unrelated page is left alone
        titles = ['2018_FIFA_World_Cup', 'Brazil', 'FIFA_World_Cup', '2022_FIFA_World_Cup', '2010_FIFA_World_Cup']
        assert families(titles) == [[0, 2, 3, 4]]
        # an umbrella shared by two templates merges them into one family (a partition)
        assert families(['2018_Open', 'Open', 'Open_2019', 'Cats']) == [[0, 1, 2]]
        assert families(['Brazil', 'FIFA_World_Cup']) == []  # no slotted member, no family
        assert filter_type(2021, 'COVID-19_pandemic_in_Spain') == 'filter-covid'
        assert filter_type(2021, '2020_Coppa_Italia_Final') == 'filter-recurrent'
        assert filter_type(2021, 'Vert_ramp') == 'filter-other'
        print('families self-check ok')
