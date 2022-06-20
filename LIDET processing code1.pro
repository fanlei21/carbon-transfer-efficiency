PRO pj2 
  dri='input data dir XXX'
  outdri2='output data dir XXX'
  data1=dri+'SITEdata.txt'
  nlines = FILE_LINES(data1)
  OPENR, lun, data1, /get_lun
  SITE=strarr(1,nlines)
  READf,lun,SITE
  FREE_LUN,lun

  data1=dri+'SPECIESdata.txt'
  OPENR, lun, data1, /get_lun
  SPECIES=strarr(1,nlines)
  READf,lun,SPECIES
  FREE_LUN,lun

  data1=dri+'TYPE1data.txt'
  OPENR, lun, data1, /get_lun
  TYPE1=strarr(1,nlines)
  READf,lun,TYPE1
  FREE_LUN,lun

  data1=dri+'BIOMEdata.txt'
  OPENR, lun, data1, /get_lun
  BIOME=strarr(1,nlines)
  READf,lun,BIOME
  FREE_LUN,lun

  data1=dri+'HLZdata.txt'
  OPENR, lun, data1, /get_lun
  HLZ=strarr(1,nlines)
  READf,lun,HLZ
  FREE_LUN,lun

  data2=dri+'data.txt'
  OPENR, lun, data2, /get_lun
  ydata=fltarr(4,nlines)
  READf,lun,ydata
  FREE_LUN,lun

  var= SITE[sort(SITE)]
  SITE2=var[uniq(var)]
  var= SPECIES[sort(SPECIES)]
  SPECIES2=var[uniq(var)]
  var= TYPE1[sort(TYPE1)]
  TYPE2=var[uniq(var)]
  var= BIOME[sort(BIOME)]
  BIOME2=var[uniq(var)]
  var= HLZ[sort(HLZ)]
  HLZ2=var[uniq(var)]
  
  temp12=outdri2+'out.txt'
  OPENw, lun, temp12, /get_lun
  for i=0,N_Elements(SITE2)-1 do begin
    for j=0,N_Elements(SPECIES2)-1 do begin
      for k=0,N_Elements(TYPE2)-1 do begin
        for l=0,N_Elements(BIOME2)-1 do begin
          for m=0,N_Elements(HLZ2)-1 do begin
            temp=SITE eq SITE2[i]
            temp2 =SPECIES eq SPECIES2[j]
            temp3 =TYPE1 eq TYPE2[k]
            temp4 =BIOME eq BIOME2[l]
            temp5 =HLZ eq HLZ2[m]
            temp6=temp and temp2 and temp3 and temp4 and temp5
            temp7 =WHERE(temp6 eq 1,cou)
            ;temp8=outdri+SITE2[i]+'_'+SPECIES2[j]+'_'+TYPE2[k]+'_'+BIOME2[l]+'_'+HLZ2[m]+'_ALL.txt'
            if cou eq 0 then begin
              continue
            endif
            ;OPENw, lun, temp8, /get_lun
            ;printf,lun,ydata[*,temp7]
            ;FREE_LUN,lun
            temp9=ydata[*,temp7]
            temp12=temp9[0,*]
            pj=mean(temp12)
            bz=stddev(temp12)
            printf,lun,pj,bz,' ',SITE2[i],' ',SPECIES2[j],' ',TYPE2[k] 
            
          endfor
        endfor
      endfor
    endfor
  endfor

FREE_LUN,lun
end