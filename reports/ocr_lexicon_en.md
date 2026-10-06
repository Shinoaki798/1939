# OCR lexicon v2, en

```json
{
 "lexicon_version": "v2",
 "lang": "en",
 "anchor": {
  "sources": [
   {
    "name": "scowl",
    "file": "scowl-2020.12.07.tar.gz",
    "sha256": "5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc",
    "lists": 38,
    "categories": [
     "english",
     "american"
    ],
    "max_size": 60,
    "types": 87580
   },
   {
    "name": "ecco_tcp",
    "files": {
     "part-0000.parquet": "59cdd3d812ea8dc1d6d33517f4173675905c7b2f26d51ddfc6ac95e1b841250d",
     "part-0001.parquet": "9be36c2e7558371bd8e4afdc64d5864fd88ba293bfcf94d2dee30aaffbd5d1f5",
     "part-0002.parquet": "7239c49155b04ac3f9be80ec232b39e7eee53813d0797b62c0bd54d7b09546b9"
    },
    "docs": 3101,
    "types_freq_ge_min": 263226
   },
   {
    "name": "evans_tcp",
    "files": {
     "part-0000.parquet": "33fb17d6f0358112cc476c33714f2eb3b1b1bdc8949ff0e926a40d303b8c64d5",
     "part-0001.parquet": "8c199bd98125d5e5defe2195da349f6dc7d161b05c88f44c2c2a157cf5d1a60e",
     "part-0002.parquet": "e68688ffd8121f6cb69ead3528e2cf79f141c6b17b1c6b2d7a3b2f9539e30fa0",
     "part-0003.parquet": "41edb1a63468713525d451c61b9052fe7d5f0ade7bcdd5317828fe1a76029427"
    },
    "docs": 5012,
    "types_freq_ge_min": 160416
   },
   {
    "name": "tcp_union",
    "types": 334234
   }
  ]
 },
 "anchor_types": 370774,
 "pool": {
  "sources": [
   "american_stories"
  ],
  "files": 56,
  "docs_sampled": 285717,
  "sampling": {
   "method": "article-id hash",
   "rate": 0.0015933457873792492
  },
  "date_range": [
   "1900-01-01",
   "1939-06-30"
  ],
  "min_df": 50,
  "min_titles": 10
 },
 "pool_types_before_filter": 24572,
 "pool_types_in_anchor": 23557,
 "variant_filter": {
  "max_edit_distance": 1,
  "ratio": 50,
  "applies_to": "pool types not in the anchor",
  "dropped_types": 275,
  "dropped_file": "variants_dropped_52eef5d79ce7.tsv"
 },
 "pool_types_after_filter": 24297,
 "pool_only_types_after_filter": 740,
 "total_types": 371514,
 "file": "52eef5d79ce7.txt",
 "sha256": "52eef5d79ce770764e7d73b50dcbe50e8b9791e6688e85c5631d33627fd48802",
 "built_at": "2026-10-06T04:28:42+00:00"
}
```

## Variant filter: top 50 dropped pool types by frequency

| type | freq in pool sample | more frequent neighbour | its freq | ratio |
|---|---|---|---|---|
| lhe | 503 | the | 2,762,673 | 5492x |
| tlon | 339 | ton | 28,621 | 84x |
| shs | 304 | she | 62,163 | 204x |
| aas | 276 | was | 319,496 | 1158x |
| srs | 252 | mrs | 130,245 | 517x |
| thls | 247 | this | 188,860 | 765x |
| homa | 243 | home | 52,195 | 215x |
| lne | 238 | one | 115,191 | 484x |
| mako | 219 | make | 31,140 | 142x |
| iof | 216 | of | 1,535,656 | 7110x |
| sll | 209 | all | 117,785 | 564x |
| ihs | 200 | is | 412,439 | 2062x |
| hne | 179 | he | 228,694 | 1278x |
| waa | 178 | was | 319,496 | 1795x |
| fne | 176 | one | 115,191 | 654x |
| aii | 175 | ii | 83,558 | 477x |
| sz | 173 | so | 68,479 | 396x |
| nve | 169 | ne | 11,633 | 69x |
| thst | 167 | that | 353,603 | 2117x |
| fnr | 166 | for | 402,244 | 2423x |
| ofice | 165 | office | 22,608 | 137x |
| ssc | 165 | ss | 8,963 | 54x |
| ioi | 164 | ii | 83,558 | 510x |
| inr | 159 | in | 786,212 | 4945x |
| rcd | 156 | red | 9,391 | 60x |
| ssd | 151 | ss | 8,963 | 59x |
| ioo | 141 | too | 13,947 | 99x |
| fss | 138 | ss | 8,963 | 65x |
| sucn | 138 | such | 32,130 | 233x |
| presl | 137 | press | 7,635 | 56x |
| speciad | 135 | special | 15,092 | 112x |
| avs | 134 | as | 207,340 | 1547x |
| ceme | 132 | come | 19,783 | 150x |
| hcr | 130 | her | 89,648 | 690x |
| oeen | 129 | been | 120,758 | 936x |
| ssr | 129 | ss | 8,963 | 69x |
| citv | 125 | city | 69,242 | 554x |
| hsr | 125 | her | 89,648 | 717x |
| ssn | 125 | son | 19,831 | 159x |
| cnt | 123 | cent | 16,904 | 137x |
| flrst | 123 | first | 50,348 | 409x |
| fthe | 122 | the | 2,762,673 | 22645x |
| jy | 122 | by | 241,027 | 1976x |
| tbo | 121 | to | 870,123 | 7191x |
| zc | 121 | sc | 7,132 | 59x |
| cne | 120 | one | 115,191 | 960x |
| iio | 120 | ii | 83,558 | 696x |
| thnt | 120 | that | 353,603 | 2947x |
| madi | 119 | made | 60,505 | 508x |
| dsy | 116 | day | 66,520 | 573x |
