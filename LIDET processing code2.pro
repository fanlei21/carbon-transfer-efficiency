PRO hbpj2  
  dri='input data dir XXX'
  data1='input data XXX'  £»txt file
  outdri2='output data XXX'  £»txt file

  nlines = FILE_LINES(data1)
  OPENR, lun, data1, /get_lun
  SITE=strarr(1,nlines)
  READf,lun,SITE
  FREE_LUN,lun
  
  OPENw, lun2, outdri2, /get_lun,WIDTH=3000
  for i=0,nlines-1 do begin
    aa=strsplit(SITE[i],' ',/extract)
    
    file_path = FILE_SEARCH(dri,aa[2]+'_'+aa[3]+'_'+aa[4]+'*.txt',COUNT=shul,/test_regular)
    if shul eq 0 then begin
      continue
    endif
    hs = FILE_LINES(file_path[0])
    OPENR, lun, file_path[0], /get_lun
    ydata=fltarr(3,hs)
    READf,lun,ydata
    FREE_LUN,lun
    ydata1=transpose(ydata)
    ydata1[*,0]=round(ydata1[*,0]*365)
    
    bb=reform(ydata1,3*hs,1)
    
    printf,lun2,aa[2],' ',aa[3],' ',aa[4],' ',aa[0],' ',aa[1],' ',hs,' ',bb
    
  endfor
  
  FREE_LUN,lun2
  
end