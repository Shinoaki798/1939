# OCR lexicon v2, de

```json
{
 "lexicon_version": "v2",
 "lang": "de",
 "anchor": {
  "sources": [
   {
    "name": "dta",
    "file": "dta_komplett_2026-02-10.zip",
    "sha256": "76d50522ffb6ece63991dc145c17a229fec4623179c6fb7ec95d3be75217ad35",
    "docs_used": 5465,
    "docs_skipped_after_1938_or_undated": 16,
    "types": 1315216
   }
  ]
 },
 "anchor_types": 1315216,
 "pool": {
  "sources": [
   "ddb_newspapers_de",
   "europeana_newspapers_de",
   "voelkischer_beobachter_de"
  ],
  "files": 647,
  "docs_sampled": 179028,
  "sampling": {
   "method": "first in-range documents per file",
   "per_file": 1500
  },
  "date_range": [
   "1900-01-01",
   "1939-06-30"
  ],
  "min_df": 50,
  "min_titles": 5
 },
 "pool_types_before_filter": 184802,
 "pool_types_in_anchor": 135218,
 "variant_filter": {
  "max_edit_distance": 1,
  "ratio": 50,
  "applies_to": "pool types not in the anchor",
  "dropped_types": 19629,
  "dropped_file": "variants_dropped_e04cdd822061.tsv"
 },
 "pool_types_after_filter": 165173,
 "pool_only_types_after_filter": 29955,
 "total_types": 1345171,
 "file": "e04cdd822061.txt",
 "sha256": "e04cdd8220615291cbff602d7c33a2b331f1d29607833a32773d0a59749ad5cd",
 "built_at": "2026-10-06T04:26:02+00:00"
}
```

## Variant filter: top 50 dropped pool types by frequency

| type | freq in pool sample | more frequent neighbour | its freq | ratio |
|---|---|---|---|---|
| unä | 19,013 | und | 8,332,443 | 438x |
| esellschaft | 5,771 | gesellschaft | 461,323 | 80x |
| äo | 3,712 | so | 842,421 | 227x |
| iür | 3,472 | für | 2,223,766 | 640x |
| unü | 3,335 | und | 8,332,443 | 2498x |
| öer | 3,298 | der | 11,802,116 | 3579x |
| äie | 3,276 | die | 10,062,591 | 3072x |
| unö | 3,260 | und | 8,332,443 | 2556x |
| mtsgericht | 3,211 | amtsgericht | 365,003 | 114x |
| aufmann | 3,177 | kaufmann | 216,967 | 68x |
| amtegericht | 2,933 | amtsgericht | 365,003 | 124x |
| vcr | 2,879 | vor | 777,851 | 270x |
| udr | 2,875 | uhr | 879,596 | 306x |
| ührer | 2,806 | ihrer | 178,910 | 64x |
| auguft | 2,803 | august | 237,976 | 85x |
| üer | 2,701 | der | 11,802,116 | 4370x |
| erv | 2,690 | er | 1,813,229 | 674x |
| äen | 2,609 | den | 4,359,431 | 1671x |
| esucht | 2,585 | gesucht | 168,893 | 65x |
| vsr | 2,582 | vor | 777,851 | 301x |
| wk | 2,513 | mk | 296,707 | 118x |
| ztm | 2,445 | zum | 877,947 | 359x |
| äe | 2,404 | ge | 649,060 | 270x |
| erstr | 2,370 | erst | 124,519 | 53x |
| nachn | 2,325 | nach | 1,200,208 | 516x |
| jür | 2,175 | für | 2,223,766 | 1022x |
| jein | 2,135 | ein | 1,938,550 | 908x |
| äi | 2,078 | ii | 307,678 | 148x |
| pfb | 2,064 | pf | 107,286 | 52x |
| aktten | 2,019 | aktien | 242,016 | 120x |
| äes | 2,006 | des | 3,452,460 | 1721x |
| attien | 1,982 | aktien | 242,016 | 122x |
| aufw | 1,979 | auf | 2,328,233 | 1176x |
| nhr | 1,973 | uhr | 879,596 | 446x |
| ausk | 1,922 | aus | 1,467,185 | 763x |
| vdr | 1,869 | vor | 777,851 | 416x |
| inbaber | 1,867 | inhaber | 118,531 | 63x |
| amisgericht | 1,854 | amtsgericht | 365,003 | 197x |
| cee | 1,822 | ee | 128,005 | 70x |
| iind | 1,820 | sind | 931,708 | 512x |
| uü | 1,810 | um | 784,238 | 433x |
| jnli | 1,795 | juli | 258,987 | 144x |
| ofort | 1,768 | sofort | 193,077 | 109x |
| bci | 1,754 | bei | 1,458,932 | 832x |
| gpf | 1,733 | pf | 107,286 | 62x |
| gewiun | 1,723 | gewinn | 112,366 | 65x |
| ktien | 1,716 | aktien | 242,016 | 141x |
| deö | 1,710 | der | 11,802,116 | 6902x |
| beschräukter | 1,702 | beschränkter | 123,388 | 72x |
| dsn | 1,669 | den | 4,359,431 | 2612x |
