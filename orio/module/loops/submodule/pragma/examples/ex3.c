/* This example just enumerates all possible modifiers of vector pragma.
 * Each pragma is mutually exclusive.
 */
void ex3(int n, double ss, double* a, double* b, double* y) {

/*@ Loops(transform Pragma(pragma_str=["vector always",
                                       "vector always assert",
                                       "vector aligned",
                                       "vector unaligned",
                                       "vector temporal",
                                       "vector nontemporal",
                                       "vector nontemporal (a,b,y)",
                                       "novector"
                                       ]))
@*/

  for(i=0; i<=n; i++) {
    y[i] = b[i] + ss*a[i];
  }

/*@ @*/
}
