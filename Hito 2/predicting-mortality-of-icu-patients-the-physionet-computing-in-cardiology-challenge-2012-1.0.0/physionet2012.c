/* file: physionet2012.c	G. Moody	22 March 2012

Sample entry for the PhysioNet/CinC Challenge 2012

This sample requires saps_score.c.  Compile it using gcc by
    gcc -o physionet2012 physionet2012.c saps_score.c
*/

#include <stdio.h>
#include <string.h>
#define PLEN 16	/* max length of data type string, including trailing zero */

struct data {
    int time;   	/* minutes since admission */
    char p[PLEN];	/* type of data */
    double val; 	/* parameter value */
};

/* Risk of in-hospital death as a function of calculated SAPS score [0-56].
   These values are *examples*;  they are *not* good estimates of risk! */
double risk[] = {			/*  SAPS */
    .1, .1, .1, .1, .1, .1, .1, .1,	/*  0- 7 */
    .1, .1, .1, .1, .1, .1, .1, .1,	/*  8-15 */
    .1, .1, .2, .2, .3, .3, .3, .4,	/* 16-23 */
    .4, .4, .5, .5, .5, .6, .6, .6,	/* 24-31 */
    .7, .7, .7, .8, .8, .9, .9, .9,	/* 32-39 */
    .9, .9, .9, .9, .9, .9, .9, .9,	/* 40-47 */
    .9, .9, .9, .9, .9, .9, .9, .9, .9	/* 48-56 */
};

main(int argc, char **argv)
{
    char buf[128], RecordID[20], pname[PLEN], *s, *t;
    double value;
    int h, m, n = 0, saps, survival;
    static struct data Data[2000], *d = Data;

    /* Look for the RecordID. */
    while (fgets(buf, sizeof(buf), stdin)) {
	if (strncmp(buf, "00:00,RecordID,", 15) == 0) {
	    strncpy(RecordID, buf+15, sizeof(RecordID));
	    RecordID[strlen(RecordID)-1] = '\0';  /* discard trailing newline */
	    break;
	}
    }

    /* Read the remainder of the input, parse it, store into Data */
    while (fgets(buf, sizeof(buf), stdin)) {
	for (s = buf; *s && *s != ','; s++)
	    ;
	if (*s == '\0') continue; /* skip this line if no comma found */
	*s++ = '\0';
	if (sscanf(buf, "%d:%d", &h, &m) == 2) d->time = 60*h + m;
	else continue;		  /* skip if first field not in hh:mm form */
	for (t = s; *t && *t != ','; t++)
	    ;
	if (*t == '\0') continue; /* skip if second comma is missing */
	*t++ = '\0';
	strncpy(d->p, s, PLEN);
	if (sscanf(t, "%lf", &value) == 1) {
	    d->val = value;
	    d++;
	    n++;
	}
    }

    /* Analyse the data.  This sample entry calculates SAPS-I from the data. */
    saps = saps_score(&Data, n);
    
    /* The saps_score function calculates a score even if one or more SAPS
       parameters are missing, but returns a negative score in this case.
       Here, we treat a negative (truncated) score as if it were positive. */
   if (saps < 0) saps = -saps;
 
   /* This sample entry predicts death (survival = 1) if SAPS is 20 or more. */
   survival = (saps >= 20);

   /* It uses risk[saps] to estimate risk for event 2. */
   printf("%s,%d,%g\n", RecordID, survival, risk[saps]);
}
