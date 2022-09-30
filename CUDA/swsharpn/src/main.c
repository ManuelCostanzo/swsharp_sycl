#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define SYCL_LANGUAGE_VERSION 1
#include "swsharp/swsharp.h"

static FILE *fileSafeOpen(const char *path, const char *mode)
{
  FILE *f = fopen(path, mode);
  return f;
}

double dwalltime()
{
  double sec;
  struct timeval tv;

  gettimeofday(&tv, NULL);
  sec = tv.tv_sec + tv.tv_usec / 1000000.0;
  return sec;
}

#define ASSERT(expr, fmt, ...)                              \
  do                                                        \
  {                                                         \
    if (!(expr))                                            \
    {                                                       \
      fprintf(stderr, "[ERROR]: " fmt "\n", ##__VA_ARGS__); \
      exit(-1);                                             \
    }                                                       \
  } while (0)

#define CHAR_INT_LEN(x) (sizeof(x) / sizeof(CharInt))

typedef struct CharInt
{
  const char *format;
  const int code;
} CharInt;

static struct option options[] = {{"query", required_argument, 0, 'i'},
                                  {"target", required_argument, 0, 'j'},
                                  {"gap-open", required_argument, 0, 'g'},
                                  {"gap-extend", required_argument, 0, 'e'},
                                  {"match", required_argument, 0, 'a'},
                                  {"mismatch", required_argument, 0, 'b'},
                                  {"score", no_argument, 0, 's'},
                                  {"cards", required_argument, 0, 'c'},
                                  {"out", required_argument, 0, 'o'},
                                  {"outfmt", required_argument, 0, 't'},
                                  {"algorithm", required_argument, 0, 'A'},
                                  {"cpu", no_argument, 0, 'P'},
                                  {"help", no_argument, 0, 'h'},
                                  {0, 0, 0, 0}};

static CharInt outFormats[] = {{"pair", SW_OUT_PAIR},
                               {"pair-stat", SW_OUT_STAT_PAIR},
                               {"plot", SW_OUT_PLOT},
                               {"stat", SW_OUT_STAT},
                               {"dump", SW_OUT_DUMP}};

static CharInt algorithms[] = {
    {"SW", SW_ALIGN}, {"NW", NW_ALIGN}, {"HW", HW_ALIGN}, {"OV", OV_ALIGN}};

static void getCudaCards(int **cards, int *cardsLen, char *optarg);

static int getOutFormat(char *optarg);
static int getAlgorithm(char *optarg);

static void help();

int main(int argc, char *argv[])
{

  char *queryPath = NULL;
  char *targetPath = NULL;

  int match = 1;
  int mismatch = -3;

  int gapOpen = 5;
  int gapExtend = 2;

  int scoreOnly = 0;

  int cardsLen = -1;
  int *cards = NULL;

  char *out = NULL;
  int outFormat = SW_OUT_STAT_PAIR;

  int algorithm = SW_ALIGN;

  int forceCpu = 0;

  while (1)
  {

    char argument = getopt_long(argc, argv, "i:j:g:e:h", options, NULL);

    if (argument == -1)
    {
      break;
    }

    switch (argument)
    {
    case 'i':
      queryPath = optarg;
      break;
    case 'j':
      targetPath = optarg;
      break;
    case 'g':
      gapOpen = atoi(optarg);
      break;
    case 'e':
      gapExtend = atoi(optarg);
      break;
    case 'a':
      match = atoi(optarg);
      break;
    case 'b':
      mismatch = atoi(optarg);
      break;
    case 's':
      scoreOnly = 1;
      break;
    case 'c':
      getCudaCards(&cards, &cardsLen, optarg);
      break;
    case 'o':
      out = optarg;
      break;
    case 't':
      outFormat = getOutFormat(optarg);
      break;
    case 'A':
      algorithm = getAlgorithm(optarg);
      break;
    case 'P':
      forceCpu = 1;
      break;
    case 'h':
    default:
      help();
      return -1;
    }
  }

  ASSERT(queryPath != NULL, "missing option -i (query file)");
  ASSERT(targetPath != NULL, "missing option -j (target file)");

  if (forceCpu)
  {
    cards = NULL;
    cardsLen = 0;
  }
  else
  {

    if (cardsLen == -1)
    {
      cudaGetCards(&cards, &cardsLen);
    }

    ASSERT(cudaCheckCards(cards, cardsLen), "invalid cuda cards");
  }

  double time = dwalltime();

  Scorer *scorer;
  scorerCreateScalar(&scorer, match, mismatch, gapOpen, gapExtend);

  Chain *query = NULL;
  Chain *target = NULL;

  readFastaChain(&query, queryPath);
  readFastaChain(&target, targetPath);

  threadPoolInitialize(1);

  if (scoreOnly)
  {

    int score;
    scorePair(&score, algorithm, query, target, scorer, cards, cardsLen, NULL);

    time = dwalltime() - time;

    outputScore(score, query, target, scorer, out);
  }
  else
  {

    Alignment *alignment;
    alignPair(&alignment, algorithm, query, target, scorer, cards, cardsLen,
              NULL);

    ASSERT(checkAlignment(alignment), "invalid align");

    time = dwalltime() - time;

    outputAlignment(alignment, out, outFormat);

    alignmentDelete(alignment);
  }

  FILE *file = out == NULL ? stdout : fileSafeOpen(out, "w");

  // fprintf(
  //     file, "SW#N;%s;%s;%d;%d;%d;%d;%d,%lf;%lf\n", queryPath, targetPath,
  //     gapOpen, gapExtend, cardsLen, chainGetLength(query),
  //     chainGetLength(target), time,
  //     ((long int)(chainGetLength(query)) * (long int)(chainGetLength(target))) /
  //         (time * 1000000000));

  printf("TIME: %lf\n", time);

  chainDelete(query);
  chainDelete(target);

  scorerDelete(scorer);

  threadPoolTerminate();
  free(cards);

  return 0;
}

static void getCudaCards(int **cards, int *cardsLen, char *optarg)
{

  *cardsLen = strlen(optarg);
  *cards = (int *)malloc(*cardsLen * sizeof(int));

  int i;
  for (i = 0; i < *cardsLen; ++i)
  {
    (*cards)[i] = optarg[i] - '0';
  }
}

static int getOutFormat(char *optarg)
{

  int i;
  for (i = 0; i < CHAR_INT_LEN(outFormats); ++i)
  {
    if (strcmp(outFormats[i].format, optarg) == 0)
    {
      return outFormats[i].code;
    }
  }

  ASSERT(0, "unknown out format %s", optarg);
}

static int getAlgorithm(char *optarg)
{

  int i;
  for (i = 0; i < CHAR_INT_LEN(algorithms); ++i)
  {
    if (strcmp(algorithms[i].format, optarg) == 0)
    {
      return algorithms[i].code;
    }
  }

  ASSERT(0, "unknown algorithm %s", optarg);
}

static void help()
{
  printf(
      "usage: swsharpn -i <query file> -j <target file> [arguments ...]\n"
      "\n"
      "arguments:\n"
      "    -i, --query <file>\n"
      "        (required)\n"
      "        input fasta query file\n"
      "    -j, --target <file>\n"
      "        (required)\n"
      "        input fasta target file\n"
      "    -g, --gap-open <int>\n"
      "        default: 5\n"
      "        gap opening penalty, must be given as a positive integer \n"
      "    -e, --gap-extend <int>\n"
      "        default: 1\n"
      "        gap extension penalty, must be given as a positive integer and\n"
      "        must be less or equal to gap opening penalty\n"
      "    --match <int>\n"
      "        default: 1\n"
      "        match score\n"
      "    --mismatch <int>\n"
      "        default: -3\n"
      "        mismatch score\n"
      "    --algorithm <string>\n"
      "        default: SW\n"
      "        algorithm used for alignment, must be one of the following: \n"
      "            SW - Smith-Waterman local alignment\n"
      "            NW - Needleman-Wunsch global alignment\n"
      "            HW - semiglobal alignment\n"
      "            OV - overlap alignment\n"
      "    --cards <ints>\n"
      "        default: all available CUDA cards\n"
      "        list of cards should be given as an array of card indexes "
      "delimited with\n"
      "        nothing, for example usage of first two cards is given as "
      "--cards 01\n"
      "    --score\n"
      "        if given only the score without the alignment is outputed, "
      "also\n"
      "        --outfmt option is ignored\n"
      "    --out <string>\n"
      "        default: stdout\n"
      "        output file for the alignment\n"
      "    --outfmt <string>\n"
      "        default: pair-stat\n"
      "        out format for the output file, must be one of the following: \n"
      "            pair      - emboss pair output format \n"
      "            pair-stat - combination of pair and stat output\n"
      "            plot      - output used for plotting alignment with gnuplot "
      "\n"
      "            stat      - statistics of the alignment\n"
      "            dump      - binary format for usage with swsharpout\n"
      "    --cpu\n"
      "        only cpu is used\n"
      "    -h, -help\n"
      "        prints out the help\n");
}
