/* file: saps_score.c		G. Moody	22 March 2012

Calculate SAPS-I score given data from a Challenge 2012 record
*/

#define PLEN 16	/* max length of data type string, including trailing zero */
struct data {
    int time;   	/* minutes since admission */
    char p[PLEN];		/* type of data */
    double val;		/* parameter value */
};

typedef struct data Data; 

#define MISSING (-2)
#define INVALID (-1)


int score_age(double age)
{
    if     (age > 200) return INVALID;
    else if (age > 75) return 4;
    else if (age > 65) return 3;
    else if (age > 55) return 2;
    else if (age > 45) return 1;
    else if (age >= 0) return 0;
    else               return MISSING;
}

int score_hr(double hr)
{
    if      (hr >  250) return INVALID;
    else if (hr >= 180) return 4;
    else if (hr >= 140) return 3;
    else if (hr >= 110) return 2;
    else if (hr >=  70) return 0;
    else if (hr >=  55) return 2;
    else if (hr >=  40) return 3;
    else if (hr >=  10) return 4;
    else if (hr >=   0) return INVALID;
    else                return MISSING;
}

int score_sbp(double sbp)
{
    if      (sbp >  300) return INVALID;
    else if (sbp >= 190) return 4;
    else if (sbp >= 150) return 2;
    else if (sbp >=  80) return 0;
    else if (sbp >=  55) return 2;
    else if (sbp >=  20) return 4;
    else if (sbp >=   0) return INVALID;
    else                 return MISSING;
}

int score_temp(double temp)
{
    if      (temp >  45) return INVALID;
    else if (temp >= 41) return 4;
    else if (temp >= 39) return 3;
    else if (temp >= 38.5) return 2;
    else if (temp >= 36) return 0;
    else if (temp >= 34) return 1;
    else if (temp >= 32) return 2;
    else if (temp >= 30) return 3;
    else if (temp >= 15) return 4;
    else if (temp >=  0) return INVALID;
    else		 return MISSING;
}

int score_resp(double resp)
{
    if      (resp >  80) return INVALID;
    else if (resp >= 50) return 4;
    else if (resp >= 35) return 3;
    else if (resp >= 25) return 1;
    else if (resp >= 12) return 0;
    else if (resp >= 10) return 1;
    else if (resp >=  6) return 2;
    else if (resp >=  2) return 4;
    else if (resp >=  0) return INVALID;
    else	 	 return MISSING;
}

int score_ur(double ur)
{
    if      (ur > 20.0) return INVALID;
    else if (ur >= 5.0) return 2;
    else if (ur >= 3.5) return 1;
    else if (ur >= 0.7) return 0;
    else if (ur >= 0.5) return 2;
    else if (ur >= 0.2) return 3;
    else if (ur >=   0) return 4;
    else	        return MISSING;
}

int score_bun(double bun)
{
    if      (bun >  100) return INVALID;
    else if (bun >=  55) return 4;
    else if (bun >=  36) return 3;
    else if (bun >=  29) return 2;
    else if (bun >= 7.5) return 1;
    else if (bun >= 3.5) return 0;
    else if (bun >=   1) return 1;
    else if (bun >=   0) return INVALID;
    else	 	 return MISSING;
}

int score_hct(double hct)
{
    if      (hct >  80) return INVALID;
    else if (hct >= 60) return 4;
    else if (hct >= 50) return 2;
    else if (hct >= 46) return 1;
    else if (hct >= 30) return 0;
    else if (hct >= 20) return 2;
    else if (hct >=  5) return 4;
    else if (hct >=  0) return INVALID;
    else	 	return MISSING;
}

int score_wbc(double wbc)
{
    if      (wbc > 200) return INVALID;
    else if (wbc >= 40) return 4;
    else if (wbc >= 20) return 2;
    else if (wbc >= 15) return 1;
    else if (wbc >=  3) return 0;
    else if (wbc >=  1) return 2;
    else if (wbc >=0.1) return 4;
    else if (wbc >=  0) return INVALID;
    else	        return MISSING;
}

int score_glu(double glu)
{
    if      (glu >  1000) return INVALID;
    else if (glu >= 44.5) return 4;
    else if (glu >= 27.8) return 3;
    else if (glu >= 14.0) return 1;
    else if (glu >=  3.9) return 0;
    else if (glu >=  2.8) return 2;
    else if (glu >=  1.6) return 3;
    else if (glu >=  0.5) return 4;
    else if (glu >=    0) return INVALID;
    else	  	  return MISSING;
}

int score_k(double k)
{
    if      (k >   20) return INVALID;
    else if (k >= 7.0) return 4;
    else if (k >= 6.0) return 3;
    else if (k >= 5.5) return 2;
    else if (k >= 3.5) return 0;
    else if (k >= 3.0) return 1;
    else if (k >= 2.5) return 2;
    else if (k >= 0.5) return 4;
    else if (k >=   0) return INVALID;
    else	       return MISSING;
}

int score_na(double na)
{
    if      (na >  200) return INVALID;
    else if (na >= 180) return 4;
    else if (na >= 161) return 3;
    else if (na >= 156) return 2;
    else if (na >= 151) return 1;
    else if (na >= 130) return 0;
    else if (na >= 120) return 2;
    else if (na >= 110) return 3;
    else if (na >=  50) return 4;
    else if (na >=  0)  return INVALID;
    else	 	return MISSING;
}

int score_hco3(double hco3)
{
    if      (hco3 > 100) return INVALID;
    else if (hco3 >= 40) return 4;
    else if (hco3 >= 30) return 1;
    else if (hco3 >= 20) return 0;
    else if (hco3 >= 10) return 1;
    else if (hco3 >=  5) return 2;
    else if (hco3 >=  2) return 4;
    else if (hco3 >=  0) return INVALID;
    else	 	 return MISSING;
}

int score_gcs(double gcs)
{
    if      (gcs >  15) return INVALID;
    else if (gcs >= 13) return 0;
    else if (gcs >= 10) return 1;
    else if (gcs >=  7) return 2;
    else if (gcs >=  4) return 3;
    else if (gcs >=  3) return 4;
    else if (gcs >=  0) return INVALID;
    else	 	return MISSING;
}

int saps_score(Data *d, int n)
{
    int s_age, s_hr, s_sbp, s_temp, s_resp, s_ur, s_bun, s_hct,
	s_wbc, s_glu, s_k, s_na, s_hco3, s_gcs;
    int i, s, saps, missing = 0;
    double urine_tot = 0.;
    
    s_age = s_hr = s_sbp = s_temp = s_resp = s_ur = s_bun = s_hct =
	    s_wbc = s_glu = s_k = s_na = s_hco3 = s_gcs = MISSING;

     /* examine data from the first 24 hours only */
    for (i = 0; i < n && d->time <= 24*60; i++, d++) {
	if (strcmp(d->p, "Age") == 0) {
	    if ((s = score_age(d->val)) > s_age)       s_age = s; }
	else if (strcmp(d->p, "HR") == 0) {
	    if ((s = score_hr(d->val)) > s_hr)         s_hr = s; }
	else if (strcmp(d->p, "SysABP") == 0 || strcmp(d->p, "NISysABP") == 0) {
	    if ((s = score_sbp(d->val)) > s_sbp)       s_sbp = s; }
	else if (strcmp(d->p, "Temp") == 0) {
	    if ((s = score_temp(d->val)) > s_temp)     s_temp = s; }
	else if (strcmp(d->p, "RespRate") == 0) {
	    if ((s = score_resp(d->val)) > s_resp)     s_resp = s; }
	else if (strcmp(d->p, "MechVent") == 0) {
	    if ((s = score_resp(49.0)) > s_resp)       s_resp = s; }
	else if (strcmp(d->p, "Urine") == 0)
	    urine_tot += d->val;
	else if (strcmp(d->p, "BUN") == 0) {
	    if ((s = score_bun(d->val/2.8)) > s_bun)   s_bun = s; }
	else if (strcmp(d->p, "HCT") == 0) {
	    if ((s = score_hct(d->val)) > s_hct)       s_hct = s; }
	else if (strcmp(d->p, "WBC") == 0) {
	    if ((s = score_wbc(d->val)) > s_wbc)       s_wbc = s; }
	else if (strcmp(d->p, "Glucose") == 0) {
	    if ((s = score_glu(d->val/18.0)) > s_glu)  s_glu = s; }
	else if (strcmp(d->p, "K") == 0) {
	    if ((s = score_k(d->val)) > s_k)  	       s_k = s; }
	else if (strcmp(d->p, "Na") == 0) {
	    if ((s = score_na(d->val)) > s_na)         s_na = s; }
	else if (strcmp(d->p, "HCO3") == 0) {
	    if ((s = score_hco3(d->val)) > s_hco3)     s_hco3 = s; }
	else if (strcmp(d->p, "GCS") == 0) {
	    if ((s = score_gcs(d->val)) > s_gcs)       s_gcs = s; }
    }

    /* s_ur is determined by total urine output over 24 hours */
    s_ur = score_ur(urine_tot/1000.0);

    /* Sum the SAPS components. */
    if (s_age   >= 0) saps  = s_age;  else missing++;
    if (s_hr    >= 0) saps += s_hr;   else missing++;
    if (s_sbp   >= 0) saps += s_sbp;  else missing++;
    if (s_temp  >= 0) saps += s_temp; else missing++;
    if (s_resp  >= 0) saps += s_resp; else missing++;
    if (s_ur    >= 0) saps += s_ur;   else missing++;
    if (s_bun   >= 0) saps += s_bun;  else missing++;
    if (s_hct   >= 0) saps += s_hct;  else missing++;
    if (s_wbc   >= 0) saps += s_wbc;  else missing++;
    if (s_glu   >= 0) saps += s_glu;  else missing++;
    if (s_k     >= 0) saps += s_k;    else missing++;
    if (s_na    >= 0) saps += s_na;   else missing++;
    if (s_hco3  >= 0) saps += s_hco3; else missing++;
    if (s_gcs   >= 0) saps += s_gcs;  else missing++;

    if (missing == 0) return (saps);
    else              return (-saps);	/* return partial scores as negative */
}	
