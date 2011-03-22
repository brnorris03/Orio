#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

#include "decl.h"

double rtclock()
{
  struct timezone tzp;
  struct timeval tp;
  int stat;
  gettimeofday (&tp, &tzp);
  return (tp.tv_sec + tp.tv_usec*1.0e-6);
}

int main()
{
  init_input_vars();

  double annot_t_start=0, annot_t_end=0, annot_t_total=0;
  int annot_i;

  for (annot_i=0; annot_i<REPS; annot_i++)
    {
      annot_t_start = rtclock();

#ifdef DYNAMIC
int i,j;
{
  register int cbv_1;
  cbv_1=ny-1;
#pragma ivdep
#pragma vector always
  for (i=0; i<=cbv_1; i=i+1) 
    s[i]=0;
 }

{
#pragma omp parallel for private(j,i)
  for (i=0; i<=nx-13; i=i+13) {
    double scv_136, scv_137, scv_138, scv_139, scv_140, scv_141, scv_142, scv_143;
    double scv_144, scv_145, scv_146, scv_147, scv_148, scv_149, scv_150, scv_151;
    double scv_152, scv_153, scv_154, scv_155, scv_156, scv_157, scv_158, scv_159;
    double scv_160, scv_161;
    scv_136=r[(i+10)];
    scv_137=q[i];
    scv_138=r[(i+11)];
    scv_139=r[(i+4)];
    scv_140=r[(i+5)];
    scv_141=r[(i+6)];
    scv_142=r[(i+1)];
    scv_143=q[(i+4)];
    scv_144=q[(i+3)];
    scv_145=q[(i+12)];
    scv_146=q[(i+9)];
    scv_147=q[(i+10)];
    scv_148=q[(i+8)];
    scv_149=r[(i+7)];
    scv_150=r[(i+2)];
    scv_151=r[(i+3)];
    scv_152=q[(i+2)];
    scv_153=r[(i+12)];
    scv_154=q[(i+7)];
    scv_155=r[(i+9)];
    scv_156=r[i];
    scv_157=q[(i+6)];
    scv_158=q[(i+5)];
    scv_159=q[(i+11)];
    scv_160=q[(i+1)];
    scv_161=r[(i+8)];
    scv_137=0;
    scv_160=0;
    scv_152=0;
    scv_144=0;
    scv_143=0;
    scv_158=0;
    scv_157=0;
    scv_154=0;
    scv_148=0;
    scv_146=0;
    scv_147=0;
    scv_159=0;
    scv_145=0;
    for (j=0; j<=ny-8; j=j+8) {
      double scv_1, scv_2, scv_3, scv_4, scv_5, scv_6, scv_7, scv_8;
      double scv_9, scv_10, scv_11, scv_12, scv_13, scv_14, scv_15, scv_16;
      double scv_17, scv_18, scv_19, scv_20, scv_21, scv_22, scv_23, scv_24;
      double scv_25, scv_26, scv_27, scv_28, scv_29, scv_30, scv_31, scv_32;
      double scv_33, scv_34, scv_35, scv_36, scv_37, scv_38, scv_39, scv_40;
      double scv_41, scv_42, scv_43, scv_44, scv_45, scv_46, scv_47, scv_48;
      double scv_49, scv_50, scv_51, scv_52, scv_53, scv_54, scv_55, scv_56;
      double scv_57, scv_58, scv_59, scv_60, scv_61, scv_62, scv_63, scv_64;
      double scv_65, scv_66, scv_67, scv_68, scv_69, scv_70, scv_71, scv_72;
      double scv_73, scv_74, scv_75, scv_76, scv_77, scv_78, scv_79, scv_80;
      double scv_81, scv_82, scv_83, scv_84, scv_85, scv_86, scv_87, scv_88;
      double scv_89, scv_90, scv_91, scv_92, scv_93, scv_94, scv_95, scv_96;
      double scv_97, scv_98, scv_99, scv_100, scv_101, scv_102, scv_103, scv_104;
      double scv_105, scv_106, scv_107, scv_108, scv_109, scv_110, scv_111, scv_112;
      double scv_113, scv_114, scv_115, scv_116, scv_117, scv_118, scv_119, scv_120;
      scv_1=A[(i+7)*ny+j+7];
      scv_2=A[(i+11)*ny+j+1];
      scv_3=A[(i+10)*ny+j+3];
      scv_4=A[(i+6)*ny+j+1];
      scv_5=A[(i+1)*ny+j+6];
      scv_6=A[(i+9)*ny+j+6];
      scv_7=A[i*ny+j+6];
      scv_8=A[(i+9)*ny+j+1];
      scv_9=A[(i+8)*ny+j+4];
      scv_10=A[(i+12)*ny+j+7];
      scv_11=A[(i+3)*ny+j+1];
      scv_12=A[(i+4)*ny+j+3];
      scv_13=A[i*ny+j];
      scv_14=A[(i+11)*ny+j+6];
      scv_15=s[(j+3)];
      scv_16=p[(j+7)];
      scv_17=A[(i+10)*ny+j+4];
      scv_18=A[(i+5)*ny+j+1];
      scv_19=A[(i+9)*ny+j+7];
      scv_20=A[i*ny+j+5];
      scv_21=A[(i+8)*ny+j+3];
      scv_22=s[(j+4)];
      scv_23=A[(i+11)*ny+j];
      scv_24=s[j];
      scv_25=p[(j+4)];
      scv_26=A[(i+8)*ny+j];
      scv_27=A[(i+3)*ny+j+2];
      scv_28=A[(i+4)*ny+j+2];
      scv_29=A[(i+12)*ny+j+4];
      scv_30=A[(i+9)*ny+j];
      scv_31=A[(i+2)*ny+j+3];
      scv_32=A[(i+7)*ny+j];
      scv_33=A[(i+10)*ny+j+1];
      scv_34=A[(i+7)*ny+j+5];
      scv_35=A[(i+6)*ny+j+7];
      scv_36=A[(i+12)*ny+j+1];
      scv_37=p[(j+5)];
      scv_38=p[(j+2)];
      scv_39=A[(i+9)*ny+j+3];
      scv_40=s[(j+2)];
      scv_41=A[(i+8)*ny+j+6];
      scv_42=s[(j+5)];
      scv_43=A[i*ny+j+1];
      scv_44=A[(i+2)*ny+j+6];
      scv_45=p[(j+1)];
      scv_46=A[(i+5)*ny+j+2];
      scv_47=A[(i+4)*ny+j+5];
      scv_48=A[(i+3)*ny+j+7];
      scv_49=A[(i+11)*ny+j+4];
      scv_50=A[(i+7)*ny+j+6];
      scv_51=A[(i+10)*ny+j+2];
      scv_52=A[(i+6)*ny+j+6];
      scv_53=A[(i+3)*ny+j+6];
      scv_54=A[(i+1)*ny+j+1];
      scv_55=A[i*ny+j+7];
      scv_56=A[(i+12)*ny+j+6];
      scv_57=A[(i+8)*ny+j+5];
      scv_58=A[(i+2)*ny+j+1];
      scv_59=A[(i+12)*ny+j];
      scv_60=A[(i+5)*ny+j];
      scv_61=p[(j+3)];
      scv_62=A[(i+5)*ny+j+3];
      scv_63=A[(i+4)*ny+j+4];
      scv_64=A[(i+3)*ny+j];
      scv_65=A[(i+5)*ny+j+4];
      scv_66=A[(i+1)*ny+j+2];
      scv_67=A[(i+12)*ny+j+3];
      scv_68=A[(i+1)*ny+j];
      scv_69=A[(i+7)*ny+j+3];
      scv_70=A[(i+4)*ny+j+7];
      scv_71=A[(i+9)*ny+j+5];
      scv_72=A[(i+11)*ny+j+2];
      scv_73=A[(i+2)*ny+j+4];
      scv_74=A[(i+10)*ny+j+6];
      scv_75=A[(i+6)*ny+j+5];
      scv_76=p[j];
      scv_77=A[(i+3)*ny+j+5];
      scv_78=A[(i+5)*ny+j+5];
      scv_79=p[(j+6)];
      scv_80=A[(i+7)*ny+j+4];
      scv_81=s[(j+7)];
      scv_82=A[(i+1)*ny+j+3];
      scv_83=s[(j+1)];
      scv_84=A[i*ny+j+2];
      scv_85=A[(i+9)*ny+j+2];
      scv_86=A[(i+8)*ny+j+7];
      scv_87=A[(i+6)*ny+j+4];
      scv_88=A[(i+2)*ny+j+7];
      scv_89=A[(i+11)*ny+j+5];
      scv_90=A[(i+10)*ny+j+7];
      scv_91=A[(i+4)*ny+j+6];
      scv_92=A[(i+4)*ny+j];
      scv_93=A[(i+6)*ny+j+3];
      scv_94=A[(i+1)*ny+j+4];
      scv_95=A[(i+5)*ny+j+6];
      scv_96=A[(i+9)*ny+j+4];
      scv_97=A[i*ny+j+4];
      scv_98=A[(i+7)*ny+j+1];
      scv_99=A[(i+4)*ny+j+1];
      scv_100=A[(i+3)*ny+j+3];
      scv_101=A[(i+12)*ny+j+5];
      scv_102=A[(i+6)*ny+j];
      scv_103=A[(i+1)*ny+j+7];
      scv_104=A[(i+8)*ny+j+2];
      scv_105=A[(i+2)*ny+j+2];
      scv_106=A[(i+6)*ny+j+2];
      scv_107=A[(i+5)*ny+j+7];
      scv_108=A[(i+1)*ny+j+5];
      scv_109=A[(i+2)*ny+j];
      scv_110=A[(i+10)*ny+j];
      scv_111=A[i*ny+j+3];
      scv_112=A[(i+7)*ny+j+2];
      scv_113=A[(i+3)*ny+j+4];
      scv_114=s[(j+6)];
      scv_115=A[(i+12)*ny+j+2];
      scv_116=A[(i+11)*ny+j+3];
      scv_117=A[(i+10)*ny+j+5];
      scv_118=A[(i+11)*ny+j+7];
      scv_119=A[(i+8)*ny+j+1];
      scv_120=A[(i+2)*ny+j+5];
      scv_24=scv_24+scv_156*scv_13;
      scv_24=scv_24+scv_142*scv_68;
      scv_24=scv_24+scv_150*scv_109;
      scv_24=scv_24+scv_151*scv_64;
      scv_24=scv_24+scv_139*scv_92;
      scv_24=scv_24+scv_140*scv_60;
      scv_24=scv_24+scv_141*scv_102;
      scv_24=scv_24+scv_149*scv_32;
      scv_24=scv_24+scv_161*scv_26;
      scv_24=scv_24+scv_155*scv_30;
      scv_24=scv_24+scv_136*scv_110;
      scv_24=scv_24+scv_138*scv_23;
      scv_24=scv_24+scv_153*scv_59;
      scv_83=scv_83+scv_156*scv_43;
      scv_83=scv_83+scv_142*scv_54;
      scv_83=scv_83+scv_150*scv_58;
      scv_83=scv_83+scv_151*scv_11;
      scv_83=scv_83+scv_139*scv_99;
      scv_83=scv_83+scv_140*scv_18;
      scv_83=scv_83+scv_141*scv_4;
      scv_83=scv_83+scv_149*scv_98;
      scv_83=scv_83+scv_161*scv_119;
      scv_83=scv_83+scv_155*scv_8;
      scv_83=scv_83+scv_136*scv_33;
      scv_83=scv_83+scv_138*scv_2;
      scv_83=scv_83+scv_153*scv_36;
      scv_40=scv_40+scv_156*scv_84;
      scv_40=scv_40+scv_142*scv_66;
      scv_40=scv_40+scv_150*scv_105;
      scv_40=scv_40+scv_151*scv_27;
      scv_40=scv_40+scv_139*scv_28;
      scv_40=scv_40+scv_140*scv_46;
      scv_40=scv_40+scv_141*scv_106;
      scv_40=scv_40+scv_149*scv_112;
      scv_40=scv_40+scv_161*scv_104;
      scv_40=scv_40+scv_155*scv_85;
      scv_40=scv_40+scv_136*scv_51;
      scv_40=scv_40+scv_138*scv_72;
      scv_40=scv_40+scv_153*scv_115;
      scv_15=scv_15+scv_156*scv_111;
      scv_15=scv_15+scv_142*scv_82;
      scv_15=scv_15+scv_150*scv_31;
      scv_15=scv_15+scv_151*scv_100;
      scv_15=scv_15+scv_139*scv_12;
      scv_15=scv_15+scv_140*scv_62;
      scv_15=scv_15+scv_141*scv_93;
      scv_15=scv_15+scv_149*scv_69;
      scv_15=scv_15+scv_161*scv_21;
      scv_15=scv_15+scv_155*scv_39;
      scv_15=scv_15+scv_136*scv_3;
      scv_15=scv_15+scv_138*scv_116;
      scv_15=scv_15+scv_153*scv_67;
      scv_22=scv_22+scv_156*scv_97;
      scv_22=scv_22+scv_142*scv_94;
      scv_22=scv_22+scv_150*scv_73;
      scv_22=scv_22+scv_151*scv_113;
      scv_22=scv_22+scv_139*scv_63;
      scv_22=scv_22+scv_140*scv_65;
      scv_22=scv_22+scv_141*scv_87;
      scv_22=scv_22+scv_149*scv_80;
      scv_22=scv_22+scv_161*scv_9;
      scv_22=scv_22+scv_155*scv_96;
      scv_22=scv_22+scv_136*scv_17;
      scv_22=scv_22+scv_138*scv_49;
      scv_22=scv_22+scv_153*scv_29;
      scv_42=scv_42+scv_156*scv_20;
      scv_42=scv_42+scv_142*scv_108;
      scv_42=scv_42+scv_150*scv_120;
      scv_42=scv_42+scv_151*scv_77;
      scv_42=scv_42+scv_139*scv_47;
      scv_42=scv_42+scv_140*scv_78;
      scv_42=scv_42+scv_141*scv_75;
      scv_42=scv_42+scv_149*scv_34;
      scv_42=scv_42+scv_161*scv_57;
      scv_42=scv_42+scv_155*scv_71;
      scv_42=scv_42+scv_136*scv_117;
      scv_42=scv_42+scv_138*scv_89;
      scv_42=scv_42+scv_153*scv_101;
      scv_114=scv_114+scv_156*scv_7;
      scv_114=scv_114+scv_142*scv_5;
      scv_114=scv_114+scv_150*scv_44;
      scv_114=scv_114+scv_151*scv_53;
      scv_114=scv_114+scv_139*scv_91;
      scv_114=scv_114+scv_140*scv_95;
      scv_114=scv_114+scv_141*scv_52;
      scv_114=scv_114+scv_149*scv_50;
      scv_114=scv_114+scv_161*scv_41;
      scv_114=scv_114+scv_155*scv_6;
      scv_114=scv_114+scv_136*scv_74;
      scv_114=scv_114+scv_138*scv_14;
      scv_114=scv_114+scv_153*scv_56;
      scv_81=scv_81+scv_156*scv_55;
      scv_81=scv_81+scv_142*scv_103;
      scv_81=scv_81+scv_150*scv_88;
      scv_81=scv_81+scv_151*scv_48;
      scv_81=scv_81+scv_139*scv_70;
      scv_81=scv_81+scv_140*scv_107;
      scv_81=scv_81+scv_141*scv_35;
      scv_81=scv_81+scv_149*scv_1;
      scv_81=scv_81+scv_161*scv_86;
      scv_81=scv_81+scv_155*scv_19;
      scv_81=scv_81+scv_136*scv_90;
      scv_81=scv_81+scv_138*scv_118;
      scv_81=scv_81+scv_153*scv_10;
      scv_137=scv_137+scv_13*scv_76;
      scv_160=scv_160+scv_68*scv_76;
      scv_152=scv_152+scv_109*scv_76;
      scv_144=scv_144+scv_64*scv_76;
      scv_143=scv_143+scv_92*scv_76;
      scv_158=scv_158+scv_60*scv_76;
      scv_157=scv_157+scv_102*scv_76;
      scv_154=scv_154+scv_32*scv_76;
      scv_148=scv_148+scv_26*scv_76;
      scv_146=scv_146+scv_30*scv_76;
      scv_147=scv_147+scv_110*scv_76;
      scv_159=scv_159+scv_23*scv_76;
      scv_145=scv_145+scv_59*scv_76;
      scv_137=scv_137+scv_43*scv_45;
      scv_160=scv_160+scv_54*scv_45;
      scv_152=scv_152+scv_58*scv_45;
      scv_144=scv_144+scv_11*scv_45;
      scv_143=scv_143+scv_99*scv_45;
      scv_158=scv_158+scv_18*scv_45;
      scv_157=scv_157+scv_4*scv_45;
      scv_154=scv_154+scv_98*scv_45;
      scv_148=scv_148+scv_119*scv_45;
      scv_146=scv_146+scv_8*scv_45;
      scv_147=scv_147+scv_33*scv_45;
      scv_159=scv_159+scv_2*scv_45;
      scv_145=scv_145+scv_36*scv_45;
      scv_137=scv_137+scv_84*scv_38;
      scv_160=scv_160+scv_66*scv_38;
      scv_152=scv_152+scv_105*scv_38;
      scv_144=scv_144+scv_27*scv_38;
      scv_143=scv_143+scv_28*scv_38;
      scv_158=scv_158+scv_46*scv_38;
      scv_157=scv_157+scv_106*scv_38;
      scv_154=scv_154+scv_112*scv_38;
      scv_148=scv_148+scv_104*scv_38;
      scv_146=scv_146+scv_85*scv_38;
      scv_147=scv_147+scv_51*scv_38;
      scv_159=scv_159+scv_72*scv_38;
      scv_145=scv_145+scv_115*scv_38;
      scv_137=scv_137+scv_111*scv_61;
      scv_160=scv_160+scv_82*scv_61;
      scv_152=scv_152+scv_31*scv_61;
      scv_144=scv_144+scv_100*scv_61;
      scv_143=scv_143+scv_12*scv_61;
      scv_158=scv_158+scv_62*scv_61;
      scv_157=scv_157+scv_93*scv_61;
      scv_154=scv_154+scv_69*scv_61;
      scv_148=scv_148+scv_21*scv_61;
      scv_146=scv_146+scv_39*scv_61;
      scv_147=scv_147+scv_3*scv_61;
      scv_159=scv_159+scv_116*scv_61;
      scv_145=scv_145+scv_67*scv_61;
      scv_137=scv_137+scv_97*scv_25;
      scv_160=scv_160+scv_94*scv_25;
      scv_152=scv_152+scv_73*scv_25;
      scv_144=scv_144+scv_113*scv_25;
      scv_143=scv_143+scv_63*scv_25;
      scv_158=scv_158+scv_65*scv_25;
      scv_157=scv_157+scv_87*scv_25;
      scv_154=scv_154+scv_80*scv_25;
      scv_148=scv_148+scv_9*scv_25;
      scv_146=scv_146+scv_96*scv_25;
      scv_147=scv_147+scv_17*scv_25;
      scv_159=scv_159+scv_49*scv_25;
      scv_145=scv_145+scv_29*scv_25;
      scv_137=scv_137+scv_20*scv_37;
      scv_160=scv_160+scv_108*scv_37;
      scv_152=scv_152+scv_120*scv_37;
      scv_144=scv_144+scv_77*scv_37;
      scv_143=scv_143+scv_47*scv_37;
      scv_158=scv_158+scv_78*scv_37;
      scv_157=scv_157+scv_75*scv_37;
      scv_154=scv_154+scv_34*scv_37;
      scv_148=scv_148+scv_57*scv_37;
      scv_146=scv_146+scv_71*scv_37;
      scv_147=scv_147+scv_117*scv_37;
      scv_159=scv_159+scv_89*scv_37;
      scv_145=scv_145+scv_101*scv_37;
      scv_137=scv_137+scv_7*scv_79;
      scv_160=scv_160+scv_5*scv_79;
      scv_152=scv_152+scv_44*scv_79;
      scv_144=scv_144+scv_53*scv_79;
      scv_143=scv_143+scv_91*scv_79;
      scv_158=scv_158+scv_95*scv_79;
      scv_157=scv_157+scv_52*scv_79;
      scv_154=scv_154+scv_50*scv_79;
      scv_148=scv_148+scv_41*scv_79;
      scv_146=scv_146+scv_6*scv_79;
      scv_147=scv_147+scv_74*scv_79;
      scv_159=scv_159+scv_14*scv_79;
      scv_145=scv_145+scv_56*scv_79;
      scv_137=scv_137+scv_55*scv_16;
      scv_160=scv_160+scv_103*scv_16;
      scv_152=scv_152+scv_88*scv_16;
      scv_144=scv_144+scv_48*scv_16;
      scv_143=scv_143+scv_70*scv_16;
      scv_158=scv_158+scv_107*scv_16;
      scv_157=scv_157+scv_35*scv_16;
      scv_154=scv_154+scv_1*scv_16;
      scv_148=scv_148+scv_86*scv_16;
      scv_146=scv_146+scv_19*scv_16;
      scv_147=scv_147+scv_90*scv_16;
      scv_159=scv_159+scv_118*scv_16;
      scv_145=scv_145+scv_10*scv_16;
      s[(j+3)]=scv_15;
      s[(j+4)]=scv_22;
      s[j]=scv_24;
      s[(j+2)]=scv_40;
      s[(j+5)]=scv_42;
      s[(j+7)]=scv_81;
      s[(j+1)]=scv_83;
      s[(j+6)]=scv_114;
    }
    for (; j<=ny-1; j=j+1) {
      double scv_121, scv_122, scv_123, scv_124, scv_125, scv_126, scv_127, scv_128;
      double scv_129, scv_130, scv_131, scv_132, scv_133, scv_134, scv_135;
      scv_121=A[(i+7)*ny+j];
      scv_122=A[(i+1)*ny+j];
      scv_123=A[(i+6)*ny+j];
      scv_124=A[i*ny+j];
      scv_125=A[(i+4)*ny+j];
      scv_126=A[(i+2)*ny+j];
      scv_127=A[(i+10)*ny+j];
      scv_128=p[j];
      scv_129=A[(i+11)*ny+j];
      scv_130=s[j];
      scv_131=A[(i+8)*ny+j];
      scv_132=A[(i+12)*ny+j];
      scv_133=A[(i+5)*ny+j];
      scv_134=A[(i+9)*ny+j];
      scv_135=A[(i+3)*ny+j];
      scv_130=scv_130+scv_156*scv_124;
      scv_130=scv_130+scv_142*scv_122;
      scv_130=scv_130+scv_150*scv_126;
      scv_130=scv_130+scv_151*scv_135;
      scv_130=scv_130+scv_139*scv_125;
      scv_130=scv_130+scv_140*scv_133;
      scv_130=scv_130+scv_141*scv_123;
      scv_130=scv_130+scv_149*scv_121;
      scv_130=scv_130+scv_161*scv_131;
      scv_130=scv_130+scv_155*scv_134;
      scv_130=scv_130+scv_136*scv_127;
      scv_130=scv_130+scv_138*scv_129;
      scv_130=scv_130+scv_153*scv_132;
      scv_137=scv_137+scv_124*scv_128;
      scv_160=scv_160+scv_122*scv_128;
      scv_152=scv_152+scv_126*scv_128;
      scv_144=scv_144+scv_135*scv_128;
      scv_143=scv_143+scv_125*scv_128;
      scv_158=scv_158+scv_133*scv_128;
      scv_157=scv_157+scv_123*scv_128;
      scv_154=scv_154+scv_121*scv_128;
      scv_148=scv_148+scv_131*scv_128;
      scv_146=scv_146+scv_134*scv_128;
      scv_147=scv_147+scv_127*scv_128;
      scv_159=scv_159+scv_129*scv_128;
      scv_145=scv_145+scv_132*scv_128;
      s[j]=scv_130;
    }
    q[i]=scv_137;
    q[(i+4)]=scv_143;
    q[(i+3)]=scv_144;
    q[(i+12)]=scv_145;
    q[(i+9)]=scv_146;
    q[(i+10)]=scv_147;
    q[(i+8)]=scv_148;
    q[(i+2)]=scv_152;
    q[(i+7)]=scv_154;
    q[(i+6)]=scv_157;
    q[(i+5)]=scv_158;
    q[(i+11)]=scv_159;
    q[(i+1)]=scv_160;
  }
  for (i=nx-((nx-1)%13)-1; i<=nx-1; i=i+1) {
    double scv_180, scv_181;
    scv_180=q[i];
    scv_181=r[i];
    scv_180=0;
    {
      for (j=0; j<=ny-8; j=j+8) {
        double scv_162, scv_163, scv_164, scv_165, scv_166, scv_167, scv_168, scv_169;
        double scv_170, scv_171, scv_172, scv_173, scv_174, scv_175, scv_176, scv_177;
        scv_162=s[(j+4)];
        scv_163=A[i*ny+j+4];
        scv_164=A[i*ny+j+6];
        scv_165=s[(j+2)];
        scv_166=s[(j+5)];
        scv_167=A[i*ny+j+1];
        scv_168=A[i*ny+j];
        scv_169=s[(j+3)];
        scv_170=A[i*ny+j+5];
        scv_171=s[(j+7)];
        scv_172=A[i*ny+j+3];
        scv_173=A[i*ny+j+7];
        scv_174=s[j];
        scv_175=s[(j+1)];
        scv_176=s[(j+6)];
        scv_177=A[i*ny+j+2];
        scv_174=scv_174+scv_181*scv_168;
        scv_175=scv_175+scv_181*scv_167;
        scv_165=scv_165+scv_181*scv_177;
        scv_169=scv_169+scv_181*scv_172;
        scv_162=scv_162+scv_181*scv_163;
        scv_166=scv_166+scv_181*scv_170;
        scv_176=scv_176+scv_181*scv_164;
        scv_171=scv_171+scv_181*scv_173;
        scv_180=scv_180+scv_168*p[j];
        scv_180=scv_180+scv_167*p[(j+1)];
        scv_180=scv_180+scv_177*p[(j+2)];
        scv_180=scv_180+scv_172*p[(j+3)];
        scv_180=scv_180+scv_163*p[(j+4)];
        scv_180=scv_180+scv_170*p[(j+5)];
        scv_180=scv_180+scv_164*p[(j+6)];
        scv_180=scv_180+scv_173*p[(j+7)];
        s[(j+4)]=scv_162;
        s[(j+2)]=scv_165;
        s[(j+5)]=scv_166;
        s[(j+3)]=scv_169;
        s[(j+7)]=scv_171;
        s[j]=scv_174;
        s[(j+1)]=scv_175;
        s[(j+6)]=scv_176;
      }
      for (; j<=ny-1; j=j+1) {
        double scv_178, scv_179;
        scv_178=s[j];
        scv_179=A[i*ny+j];
        scv_178=scv_178+scv_181*scv_179;
        scv_180=scv_180+scv_179*p[j];
        s[j]=scv_178;
      }
    }
    q[i]=scv_180;
  }
 }


#else

      int i,j;
      {
	register int cbv_1;
	cbv_1=ny-1;
#pragma ivdep
#pragma vector always
	for (i=0; i<=cbv_1; i=i+1) 
	  s[i]=0;
      }
      {
#pragma omp parallel for private(j,i)
	for (i=0; i<=nx-12; i=i+12) {
	  q[i]=0;
	  q[(i+1)]=0;
	  q[(i+2)]=0;
	  q[(i+3)]=0;
	  q[(i+4)]=0;
	  q[(i+5)]=0;
	  q[(i+6)]=0;
	  q[(i+7)]=0;
	  q[(i+8)]=0;
	  q[(i+9)]=0;
	  q[(i+10)]=0;
	  q[(i+11)]=0;
	  register int cbv_1;
	  cbv_1=ny-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_1; j=j+1) {
	    s[j]=s[j]+r[i]*A[i][j];
	    s[j]=s[j]+r[(i+1)]*A[(i+1)][j];
	    s[j]=s[j]+r[(i+2)]*A[(i+2)][j];
	    s[j]=s[j]+r[(i+3)]*A[(i+3)][j];
	    s[j]=s[j]+r[(i+4)]*A[(i+4)][j];
	    s[j]=s[j]+r[(i+5)]*A[(i+5)][j];
	    s[j]=s[j]+r[(i+6)]*A[(i+6)][j];
	    s[j]=s[j]+r[(i+7)]*A[(i+7)][j];
	    s[j]=s[j]+r[(i+8)]*A[(i+8)][j];
	    s[j]=s[j]+r[(i+9)]*A[(i+9)][j];
	    s[j]=s[j]+r[(i+10)]*A[(i+10)][j];
	    s[j]=s[j]+r[(i+11)]*A[(i+11)][j];
	    q[i]=q[i]+A[i][j]*p[j];
	    q[(i+1)]=q[(i+1)]+A[(i+1)][j]*p[j];
	    q[(i+2)]=q[(i+2)]+A[(i+2)][j]*p[j];
	    q[(i+3)]=q[(i+3)]+A[(i+3)][j]*p[j];
	    q[(i+4)]=q[(i+4)]+A[(i+4)][j]*p[j];
	    q[(i+5)]=q[(i+5)]+A[(i+5)][j]*p[j];
	    q[(i+6)]=q[(i+6)]+A[(i+6)][j]*p[j];
	    q[(i+7)]=q[(i+7)]+A[(i+7)][j]*p[j];
	    q[(i+8)]=q[(i+8)]+A[(i+8)][j]*p[j];
	    q[(i+9)]=q[(i+9)]+A[(i+9)][j]*p[j];
	    q[(i+10)]=q[(i+10)]+A[(i+10)][j]*p[j];
	    q[(i+11)]=q[(i+11)]+A[(i+11)][j]*p[j];
	  }
	}
	for (i=nx-((nx-1)%12)-1; i<=nx-1; i=i+1) {
	  q[i]=0;
	  register int cbv_2;
	  cbv_2=ny-1;
#pragma ivdep
#pragma vector always
	  for (j=0; j<=cbv_2; j=j+1) {
	    s[j]=s[j]+r[i]*A[i][j];
	    q[i]=q[i]+A[i][j]*p[j];
	  }
	}
      }
#endif

      annot_t_end = rtclock();
      annot_t_total += annot_t_end - annot_t_start;
    }

  annot_t_total = annot_t_total / REPS;

#ifndef TEST
  printf("%f\n", annot_t_total);
#else
  {
    int i, j;
    for (i=0; i<ny; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",s[i]);
    }
    printf("\n");
    for (i=0; i<nx; i++) {
      if (i%100==0)
        printf("\n");
      printf("%f ",q[i]);
    }
  }
#endif

  return ((int) (s[0]+q[0]));

}


