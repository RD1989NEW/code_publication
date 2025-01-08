# Damit setzen wir das Working Directory auf den Ordner dieser Datei
if (!is.null(parent.frame(2)$ofile)) {
  this.dir <- dirname(parent.frame(2)$ofile)
  setwd(this.dir)
}
##inhstall required packages
install.packages("ggplot2")
#install.packages("ggsignif")
install.packages("FactoMineR")
install.packages("factoextra")
install.packages("ggplot2")
install.packages('GGally')
install.packages('Factoshiny')
install.packages("caret")
install.packages(c("ISLR", "ggplot2", "ggstatsplot","sjPlot"))
#library(ISLR)
library(ggplot2)
#library(ggstatsplot)
#library(sjPlot)
library(factoMineR)
library(factoextra)
library(data.table)
library(ggplot2)
library(caret)
library(ggplot2)
library(ggsi)
library(ggplot2)
#library(sjPlot)
library(corrplot)
library(GGally)
library(Factoshiny)
library(GGally)
library(dplyr)
library(tidyr)
library(dplyr)

print('GGplot version: ')
packageVersion("ggplot2")
print('ggsignif version: ')
packageVersion("ggsignif")
print('FactoMineR version: ')
packageVersion("FactoMineR")
print('factoextra version: ')
packageVersion("factoextra")
print('caret version: ')
packageVersion("caret")
print('corrplot version: ')
packageVersion("corrplot")
print('GGally version: ')
packageVersion("GGally")
print('data.table version: ')
packageVersion("data.table")
print('dplyr version: ')
packageVersion("dplyr")
#input file with filename: "OSAVI_Mask_ZS_Supervised_Classes_F.csv" as base input for correlation and visualization
#OSAVI_mask_Arena_Veraison<- read.table("C:/Users/Ronald/Documents/Masken_with_Supervised_Classes/OSAVI_Mask_ZS_Supervised_Classes_F.csv", header=TRUE, sep=",", row.names = NULL, check.names = FALSE)
#ALL_MASKS_MERGED_subset_1<-ALL_MASKS_MERGED[c( "W_03_08_i", "X_OSAVImean", "X_GNDVImean" ,"X_NDWImean","X_RVImean","V_OM","Diff_SAM_Bonitur" , "layer" )]
#Texture dataframe
ALL_MASKS_MERGED_T<-read.csv("C:/Users/ronal/OneDrive/Dokumente/OBIA_OSAVI_mit_Texture_NEW/OBIA_OSAVI_mit_Texture_NEW.csv", sep=',', header=TRUE)
####NEW Texture dataframe iwth all Bands from OM textures
ALL_MASKS_MERGED_T<-read.csv("C:/Users/ronal/OneDrive/Dokumente/Textur_extract/OBIA_OSAVI_MASK_extract_text_NEW.csv", sep=',', header=TRUE)

colnames(ALL_MASKS_MERGED_T)
#####noch Spalten anpassenArena_fused_data<-read.csv("C:/Users/ronal/OneDrive/Dokumente/Arena_Geodata_testsNew/Arena_Geodata_tests_new.csv",header=TRUE, sep=',')

####################################################################################################

##################################################
##subset texture dataframe
ALL_MASKS_MERGED_T_subset_1<-ALL_MASKS_MERGED_T[c( "W_03_08_i", "X_OSAVImean", "X_GNDVImean" ,"X_NDWImean","X_RVImean","X_CHM_mean", "X_CHM_stdev", "X_CHM_max","X_CHM_range","V_OM",'X_Contrastmean','X_Correlationmean', 'X_Entropymean','X_ASMmean', "X_Variance_B4mean", "X_Variance_B3mean", "X_Variance_B2mean", "X_Variance_B1mean" , "X_ASM_B1mean", "X_ASM_B2mean", "X_ASM_B3mean","X_ASM_B4mean", "X_ASM_B5mean",
                                                   "X_Contrast_B1mean", "X_Contrast_B2mean", "X_Contrast_B3mean", "X_Contrast_B4mean", "X_Correlation_B1mean", "X_Correlation_B2mean", 
                                                   "X_Correlation_B3mean", "X_Correlation_B4mean", "X_Correlation_B5mean" ,
                                                   "X_Entropy_B1mean", "X_Entropy_B2mean","X_Entropy_B3mean", "X_Entropy_B4mean", "X_Entropy_B5mean", 
                                                   "X_IDM_B1mean", "X_IDM_B2mean", "X_IDM_B3mean","X_IDM_B4mean",
                                                   "X_IDM_B5mean", "X_MOC_1_1_B_3mean", "X_MOC_1_B_1mean", "X_MOC_1_4_B_2mean", "X_MOC_2_2_B_5mean" )]


########################################

##other subset of the dataframe
ALL_MASKS_MERGED<-read.csv("D:/ALLE_MASKEN_MERGE_NEU/ALLE_MASKEN_ZUSAMMEN_MERGE_NEU_2.csv", sep=',', header=TRUE)

ALL_MASKS_MERGED_subset_1<-na.omit(ALL_MASKS_MERGED_subset_1)
corr_df<-ALL_MASKS_MERGED_subset_1
ALL_MASKS_MERGED_subset_2<-ALL_MASKS_MERGED[c( "W_03_08_i", "X_OSAVImean", "X_GNDVImean" ,"X_NDWImean","X_RVImean","V_OM","Diff_SAM_Bonitur" )]
ALL_MASKS_MERGED_subset_2_nona<-na.omit(ALL_MASKS_MERGED_subset_2)
cor(ALL_MASKS_MERGED_subset_2)
################################################################

sapply(ALL_MASKS_MERGED_subset_1, typeof)
#ALL_MASKS_MERGED_subset_1$Symptom_St_Veraison_2022<-gsub(',', '.',ALL_MASKS_MERGED_subset_1$Symptom_St_Veraison_2022 )
#ALL_MASKS_MERGED_subset_1$Symptom_St_Veraison_2022<as.numeric(ALL_MASKS_MERGED_subset_1$Symptom_St_Veraison_2022)
chars=sapply(ALL_MASKS_MERGED_subset_1, is.character)
#ALL_MASKS_MERGED_subset_1[, chars]<-as.data.frame(apply(ALL_MASKS_MERGED_subset_1[, chars], 2, as.numeric))
print(chars)
ALL_MASKS_MERGED_subset_1<-na.omit(ALL_MASKS_MERGED_subset_1)

names(ALL_MASKS_MERGED_subset_1)[2]<-"OSAVI"
names(ALL_MASKS_MERGED_subset_1)[3]<-"GNDVI"
names(ALL_MASKS_MERGED_subset_1)[4]<-"NDWI"
names(ALL_MASKS_MERGED_subset_1)[5]<-"RVI"
names(ALL_MASKS_MERGED_subset_1)[6]<-"Volume (CHM)"
names(ALL_MASKS_MERGED_subset_1)[7]<-"D_M_B(SAM)"

ALL_MASKS_MERGED_subset_x<-ALL_MASKS_MERGED[c( "W_03_08_i", "X_OSAVImean", "X_GNDVImean" ,"X_NDWImean","X_RVImean","X_CHM_mean","V_OM")]
ALL_MASKS_MERGED_subset_x<-na.omit(ALL_MASKS_MERGED_subset_x)

names(ALL_MASKS_MERGED_subset_x)[1]<-"Wuchsklasse"
names(ALL_MASKS_MERGED_subset_x)[2]<-"OSAVI"
names(ALL_MASKS_MERGED_subset_x)[3]<-"GNDVI"
names(ALL_MASKS_MERGED_subset_x)[4]<-"NDWI"
names(ALL_MASKS_MERGED_subset_x)[5]<-"RVI"
names(ALL_MASKS_MERGED_subset_x)[6]<-"CHM (m)"
names(ALL_MASKS_MERGED_subset_x)[7]<-"Volumen (CHM)"
#

#1 corr VI
ALL_MASKS_MERGED_T_subset_1_VI<-ALL_MASKS_MERGED_T[c( "W_03_08_i", "X_OSAVImean", "X_GNDVImean" ,"X_NDWImean","X_RVImean","X_TSAVImean" )]
colnames(ALL_MASKS_MERGED_T)
ALL_MASKS_MERGED_T_subset_1_VI_n<-na.omit(ALL_MASKS_MERGED_T_subset_1_VI)
cordata_VI_2=cor(ALL_MASKS_MERGED_T_subset_1_VI_n)

names(ALL_MASKS_MERGED_T_subset_1_VI)[1]<-"Growth"
names(ALL_MASKS_MERGED_T_subset_1_VI)[2]<-"OSAVI"
names(ALL_MASKS_MERGED_T_subset_1_VI)[3]<-"GNDVI"
names(ALL_MASKS_MERGED_T_subset_1_VI)[4]<-"NDWI"
names(ALL_MASKS_MERGED_T_subset_1_VI)[5]<-"RVI"
names(ALL_MASKS_MERGED_T_subset_1_VI)[6]<-"TSAVI"
cordata_df_VI_2<-as.data.frame(cordata_VI_2)
#Correlation Growth Classes with the spectral features (vegetation indices)
colnames(cordata_VI_2)<-c("Growth", "OSAVI", "GNDVI", "NDWI", "RVI", "TSAVI")
rownames(cordata_VI_2)<-c("Growth", "OSAVI", "GNDVI", "NDWI", "RVI", "TSAVI")
corrplot(cordata_VI_2, method="number")
corrplot(cordata_VI_2, method="number", type="upper")
corrplot(cordata_VI_2, method="ellipse", type="upper")

#2 corr CHM und Vol
################################################

ALL_MASKS_MERGED_T_subset_1_CHM<-ALL_MASKS_MERGED_T[c( "W_03_08_i", "X_CHM_mean", "X_CHM_stdev", "X_CHM_max","X_CHM_range","V_OM")]
ALL_MASKS_MERGED_T_subset_1_CHM_n<-na.omit(ALL_MASKS_MERGED_T_subset_1_CHM)

names(ALL_MASKS_MERGED_T_subset_1_CHM)[1]<-"Growth"
names(ALL_MASKS_MERGED_T_subset_1_CHM)[2]<-"CHM (mean)"
names(ALL_MASKS_MERGED_T_subset_1_CHM)[3]<-"CHM (std)"
names(ALL_MASKS_MERGED_T_subset_1_CHM)[4]<-"CHM (max)"
names(ALL_MASKS_MERGED_T_subset_1_CHM)[5]<-"CHM (range)"
names(ALL_MASKS_MERGED_T_subset_1_CHM)[6]<-"Volume (CHM)"

cordata_chm=cor(ALL_MASKS_MERGED_T_subset_1_CHM_n)
#correlation of the texture features with the Growth Classes
cordata_df_CHM<-as.data.frame(cordata_chm)
colnames(cordata_chm)<-c("Growth", "CHM (mean)","CHM (std)", "CHM (max)", "CHM (range)", "Volume (CHM)")
rownames(cordata_chm)<-c("Growth", "CHM (mean)","CHM (std)", "CHM (max)", "CHM (range)", "Volume (CHM)")

corrplot(cordata_chm, method="number")
corrplot(cordata_chm, method="number", type="upper")
corrplot(cordata_chm, method="ellipse", type="upper")

#corr Texture
#################

ALL_MASKS_MERGED_T_subset_1_Text<-ALL_MASKS_MERGED_T[c( "W_03_08_i", "X_Variance_B4mean", "X_Variance_B3mean", "X_Variance_B2mean", "X_Variance_B1mean" , "X_ASM_B1mean", "X_ASM_B2mean", "X_ASM_B3mean","X_ASM_B4mean", "X_ASM_B5mean",
                                                   "X_Contrast_B1mean", "X_Contrast_B2mean", "X_Contrast_B3mean", "X_Contrast_B4mean", "X_Correlation_B1mean", "X_Correlation_B2mean", 
                                                   "X_Correlation_B3mean", "X_Correlation_B4mean", "X_Correlation_B5mean" ,
                                                   "X_Entropy_B1mean", "X_Entropy_B2mean","X_Entropy_B3mean", "X_Entropy_B4mean", "X_Entropy_B5mean", 
                                                   "X_IDM_B1mean", "X_IDM_B2mean", "X_IDM_B3mean","X_IDM_B4mean",
                                                   "X_IDM_B5mean", "X_MOC_1_1_B_3mean", "X_MOC_1_B_1mean", "X_MOC_1_4_B_2mean", "X_MOC_2_2_B_5mean" )]

ALL_MASKS_MERGED_T_subset_1_Text_n<-na.omit(ALL_MASKS_MERGED_T_subset_1_Text)
cordata_text_n=cor(ALL_MASKS_MERGED_T_subset_1_Text_n)
cordata_text_n_df<-as.data.frame(as.table(cordata_text_n))
cordata_text_n_df_<-cordata_text_n_df[cordata_text_n_df$'Var2' == "W_03_08_i",]
cordata_text_n_df_$Freq<-abs(cordata_text_n_df_$Freq)
cordata_text_n_df__<-cordata_text_n_df_[cordata_text_n_df$'Freq'>0.4,]
cordata_text_n_df__na<-na.omit(cordata_text_n_df__)
colsn<-cordata_text_n_df__na[['Var1']]
ALL_MASKS_MERGED_T_subset_1_Text_n_filt<-ALL_MASKS_MERGED_T_subset_1_Text_n[, c('W_03_08_i', 'X_ASM_B1mean', 'X_ASM_B2mean','X_ASM_B3mean', 'X_ASM_B5mean', 'X_IDM_B1mean', 'X_IDM_B2mean',
                                                                                'X_IDM_B3mean', 'X_MOC_1_1_B_3mean', 'X_MOC_1_B_1mean')]
ALL_MASKS_MERGED_T_subset_1_Text_n_filt_2<-ALL_MASKS_MERGED_T_subset_1_Text_n[, colsn]


cordata_text=cor(ALL_MASKS_MERGED_T_subset_1_Text_n_filt_2)
                 
colnames(cordata_text)<-c("Growth", "ASM (B3)","IDM (B3)", "MOC (B3)","MOC (B1)")
rownames(cordata_text)<-c("Growth", "ASM (B3)","IDM (B3)", "MOC (B3)","MOC (B1)")             
                 
#correlation of the texture features with the growth classes after Porten(2020)         
cordata_df<-as.data.frame(cordata_text)

corrplot(cordata_text, method="number")
corrplot(cordata_text, method="number", type="upper")
corrplot(cordata_text, method="number", type="upper", title='Growth vs Texture', mar=c(0,0,1,0))
corrplot(cordata_text, method="ellipse", type="upper")

################################################################


ALL_MASKS_MERGED_T_subset_1_mixed<-ALL_MASKS_MERGED_T[c( "W_03_08_i", "X_OSAVImean", "X_GNDVImean" ,"X_NDWImean","X_RVImean","X_CHM_mean", "X_CHM_stdev", "X_CHM_max","X_CHM_range","V_OM",'X_Contrastmean','X_Correlationmean', 'X_Entropymean','X_ASMmean', "X_Variance_B4mean", "X_Variance_B3mean", "X_Variance_B2mean", "X_Variance_B1mean" , "X_ASM_B1mean", "X_ASM_B2mean", "X_ASM_B3mean","X_ASM_B4mean", "X_ASM_B5mean",
                                                   "X_Contrast_B1mean", "X_Contrast_B2mean", "X_Contrast_B3mean", "X_Contrast_B4mean", "X_Correlation_B1mean", "X_Correlation_B2mean", 
                                                   "X_Correlation_B3mean", "X_Correlation_B4mean", "X_Correlation_B5mean" ,
                                                   "X_Entropy_B1mean", "X_Entropy_B2mean","X_Entropy_B3mean", "X_Entropy_B4mean", "X_Entropy_B5mean", 
                                                   "X_IDM_B1mean", "X_IDM_B2mean", "X_IDM_B3mean","X_IDM_B4mean",
                                                   "X_IDM_B5mean", "X_MOC_1_1_B_3mean", "X_MOC_1_B_1mean", "X_MOC_1_4_B_2mean", "X_MOC_2_2_B_5mean" )]


cordata=cor(ALL_MASKS_MERGED_subset_x)


cordata_df<-as.data.frame(cordata)

corrplot(cordata, method="number")
corrplot(cordata, method="number", type="upper")
corrplot(cordata, method="ellipse", type="upper")
corrplot.mixed(cordata, lower="number", upper="pie",order="hclust")
#coorplot(cordata, method=)

print(cordata)

write.csv2(cordata_df, 'C:/Users/ronal/OneDrive/Dokumente/Kreuztabellen_neu/Correlations_NEW.csv')

colnames(ALL_MASKS_MERGED_T_subset_1)
ALL_MASKS_MERGED_T_subset_1<-na.omit(ALL_MASKS_MERGED_T_subset_1)
sapply(ALL_MASKS_MERGED_T_subset_1, typeof)


pca<-prcomp(ALL_MASKS_MERGED_T_subset_1[, c(2:46)], center=TRUE, scale=TRUE)
summary(pca)
str(pca)

pca$x

d2=cbind(ALL_MASKS_MERGED_T_subset_1, pca$x)
sapply(d2, typeof)
print(pca)

cordata=cor(ALL_MASKS_MERGED_T_subset_1)
corrplot(cordata)
colnames(d2)
names(d2)[1]<-"Growth"
names(d2)[2]<-"OSAVI"
names(d2)[3]<-"GNDVI"
names(d2)[4]<-"NDWI"
names(d2)[5]<-"RVI"
names(d2)[6]<-"Volume (CHM)"
names(d2)[7]<-"Contrast"
names(d2)[8]<-"Correlation"
names(d2)[9]<-"Entropy"
names(d2)[10]<-"ASM"

cordata_df<-as.data.frame(cordata)


d2$Growth<-as.factor(as.character(d2[,'Growth']))
sapply(d2, class)
ggplot(d2, aes(PC1, PC2, col=W_03_08_i, fill=W_03_08_i))+
  stat_ellipse(geom="polygon", col="red", alpha=0.5)+scale_colour_manual((name="Wuchs"),
                                                                           values=c("red", "orangered", "orange", "cyan", "green", "darkgreen"))+
  geom_point(shape=21, col='green', alpha=0.3)+theme_minimal()

colnames(d2)
rownames(d2)
print(d2$W_03_08_i)


d2_n<-d2[,-1]
sapply(d2, typeof)
sapply(d2_n, typeof)
d2$Growth<-as.factor(d2$Growth)
plot<-ggbiplot::ggbiplot(pca, varname.size = 5.0, ellipse=TRUE, circle=TRUE , circle.prob = 0.70, obs.scale=1, var.scale=1,
                   groups=d2$Growth, values=c("red"))+scale_colour_manual(
                     name="Growth Classes after Porten (2020)", values=c("red", "orangered", "orange", "cyan", "green", "darkgreen"))+
  ggtitle("PCA Growh Classes- Sensor-/ Geodata")+
  theme_minimal()+
  theme(legend.position="bottom")+theme(panel.grid.major=element_line(color="black",
                                                                      size=0.2, linetyp=1))+theme(panel.grid.minor=element_blank())

plot+coord_cartesian(xlim = c(-15,15), ylim =c(-15,15))+theme(axis.text.x=element_text(face='bold', size=15, color='black'),
                                                          axis.text.y=element_text(face='bold', size=15, color='black'))+theme(axis.title.x=element_text(size=16),
                                                                                                                               axis.text.x=element_text(size=16),
                                                                                                                               axis.title.y=element_text(size=16))+theme(plot.title=element_text(size=18))+
  theme(legend.text=element_text(size=15))+guides(color=guide_legend(override.aes = list(size=7)))


ggpairs(d2, columns=1:4, ggplot2::aes(color=Growth), title = "Growth Classes vs VI")+
  scale_fill_manual(values=c("red", "orangered", "orange", "cyan", "green", "darkgreen"))+
  scale_colour_manual(values =c("red", "orangered", "orange", "cyan", "green", "darkgreen" ))+theme(panel.grid.major=element_line(color="black",size=0.2, linetyp=1))+theme(panel.grid.minor=element_blank()+theme_bw())+theme(axis.text.x=element_text(angle=90, hjust=1))


d2_2<-d2[-c(3,5,7:15)]
names(d2_2)[1]<-'Growth'
names(d2)[1]<-'Growth'
names(d2_2)[4]<-"Volume (CHM)"
pm<-ggpairs(d2_2, columns=1:4, ggplot2::aes(color=Growth), title = "Growth Classes vs VI and Volume (CHM)")+
  scale_fill_manual(values=c("red", "orangered", "orange", "cyan", "green", "darkgreen"))+
  scale_colour_manual(values =c("red", "orangered", "orange", "cyan", "green", "darkgreen" ))+theme(panel.grid.minor=element_blank, panel.grid.major = element_blank())+theme_bw()+theme(axis.text.x=element_text(angle=90, hjust=1))

pm+theme(axis.text=element_text(size=8))+theme(strip.text.x=element_text(size=10),strip.text.y=element_text(size=10))+theme(axis.text.x=element_text(face='bold'))+theme(axis.text.y=element_text(face='bold'))+theme_bw()+theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())

OSAVI_Veraison_subset_1<-OSAVI_mask_Arena_Veraison_2[c("_NDVI_4_3m","_NDVI_5_3m", "_OSAVImean", "_GNDVImean", "_NDWImean", "_PVI_4_3me", "_RVImean", "_TSAVImean", "_CHM_mean", "W_03_08_i"  )]
OSAVI_Veraison_subset_2<-OSAVI_mask_Arena_Veraison_2[c("_NDVI_4_3m","_NDVI_5_3m", "_OSAVImean", "_GNDVImean", "_NDWImean", "_PVI_4_3me", "_RVImean", "_TSAVImean", "_CHM_mean","_PC1_SBmea", "_PC_2_SBme", "_PC_3_SBme",  "W_03_08_i"  )]


OSAVI_mask_Arena_Veraison_2<- read.csv("C:/csvUsers/Ronald/Documents/MASKEN_ZS_FINAL_/OSAVI_ZS_SC_F_2.csv", header=TRUE, sep=",", row.names = NULL, check.names = FALSE)

####################################################
##################################################

OSAVI_Veraison_subset_3<-OSAVI_mask_Arena_Veraison_2[c("_NDVI_4_3m","_NDVI_5_3m", "_OSAVImean", "_GNDVImean", "_NDWImean", "W_03_08_i", "Befall_NEU_Veraison_22",  "Symptom_St", "V_OM", "V_DTM")]
names(OSAVI_Veraison_subset_5)[1]<-"NDVI"
names(OSAVI_Veraison_subset_5)[2]<-"NDVIRE"
names(OSAVI_Veraison_subset_5)[3]<-"OSAVI"
names(OSAVI_Veraison_subset_5)[4]<-"GNDVI"
names(OSAVI_Veraison_subset_5)[5]<-"NDWI"
names(OSAVI_Veraison_subset_5)[6]<-"Wuchsklassen"
ggpairs(OSAVI_Veraison_subset_5, columns=1:6, ggplot::aes(colour=Wuchsklassen))

######################Kreuztabelle

OSAVI_Veraison_subset_KT<-OSAVI_mask_Arena_Veraison_2[c( "W_03_08_i", "Befall_NEU_Veraison_22",  "Symptom_St")]

names(OSAVI_Veraison_subset_KT)[1]<-"Wuchsklasse"
names(OSAVI_Veraison_subset_KT)[2]<-"Befallsart"
names(OSAVI_Veraison_subset_KT)[3]<-"Symptomstärke"

ALL_MASKS_MERGED_subset_2<-ALL_MASKS_MERGED[c( "W_03_08_i", "CLASS_ID_2", "CLASS_ID_3" ,"CLASS_ID_4","Diff_SAM_Bonitur", "V_OM", "layer")]

colnames(ALL_MASKS_MERGED)
ALL_MASKS_MERGED_subset_2$Growth<-as.factor(as.character(ALL_MASKS_MERGED_subset_2$W_03_08_i))
ALL_MASKS_MERGED_subset_2<-na.omit(ALL_MASKS_MERGED_subset_2)

pm<-ggpairs(ALL_MASKS_MERGED_subset_2, columns=1:5, ggplot2::aes(color=Growth), title = "Growth Classes measured vs Models")+
  scale_fill_manual(values=c("red", "orangered", "orange", "cyan", "green", "darkgreen"))+
  scale_colour_manual(values =c("red", "orangered", "orange", "cyan", "green", "darkgreen" ))+theme(panel.grid.minor=element_blank, panel.grid.major = element_blank())+theme_bw()+theme(axis.text.x=element_text(angle=90, hjust=1))

pm+theme(axis.text=element_text(size=8))+theme(strip.text.x=element_text(size=10),strip.text.y=element_text(size=10))+theme(axis.text.x=element_text(face='bold'))+theme(axis.text.y=element_text(face='bold'))+theme_bw()+theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank())


















