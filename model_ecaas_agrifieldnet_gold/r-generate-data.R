options(warn = -1)

library(plyr)
library(tidyverse)
library(raster)
library(celestial)
library(caret)
library(fastICA)
library(SOAR)
library(RStoolbox)
library(jsonlite)
library(data.table)
library(spdep)
library(FIELDimageR) ## please install via github devtools::install_github("OpenDroneMap/FIELDimageR")

####
output_path = Sys.getenv('OUTPUT_DATA')

#####
path.dir = Sys.getenv('INPUT_DATA')
data.dir = paste0(path.dir,"/ref_agrifieldnet_competition_v1")

######
label_path = "ref_agrifieldnet_competition_v1_labels_train"
test_path = "ref_agrifieldnet_competition_v1_labels_test"
image_path = "ref_agrifieldnet_competition_v1_source"

### help functions
map.func = function(x, y =7){
  map = x %>% 
    sapply(FUN = function(x){strsplit(x, '[,.:_/"]')[[1]][y]}) %>% 
    sub("[[:punct:]]", '',.) %>% 
    sub("'",'',.)
  return(map)
}


### Get the train label json file
colls =  paste0(data.dir,"/ref_agrifieldnet_competition_v1_labels_train/collection.json")
colls = jsonlite::read_json(colls)
qq = colls$links[3:length(colls$links)]
ids = c()
field_path = c()
labels_path = c()
for(i in 1:length(qq)){
  id = map.func(qq[[i]]$href) %>% as.character()
  id_path = paste0(data.dir,"/",label_path,"/",label_path,"_",id,"/field_ids.tif")
  path = paste0(data.dir,"/",label_path,"/",label_path,"_",id,"/raster_labels.tif")
  ids = c(ids,id)
  field_path = c(field_path,id_path)
  labels_path = c(labels_path,path)
}

train_path = data.frame(ids = ids, field_path = field_path,labels_path = labels_path) %>% 
  filter(ids != "windows")



#################################################
####  GETTING TRAIN DATA 7 HOURS
#### EXREACT BANDS INFO FROM IMAGES 
#################################################
df = data.frame()
for(i in 1:nrow(train_path)){
  f_mat = raster(train_path[i,2]) %>% as.matrix()
  f_mat_rs = which(rowSums(f_mat,na.rm = T)>0)
  f_mat_cs = which(colSums(f_mat,na.rm = T)>0)
  f_mat = as.matrix(f_mat[f_mat_rs,f_mat_cs])
  lab_mat = raster(train_path[i,3]) %>% as.matrix()
  lab_mat = as.matrix(lab_mat[f_mat_rs,f_mat_cs])
  

    bands = c(paste0("B0",1:9),"B11","B12")

  train_data = data.frame(id= 1)
  for(b in bands){
    img = paste0(data.dir,"/",image_path,"/",image_path,"_",train_path[i,1],"/",b,".tif")
    mm = raster(img) %>% as.matrix()
    mm = as.matrix(mm[f_mat_rs,f_mat_cs])
    
    
    bb = c()
    for(x in 1:nrow(mm)){
      for(y in 1:ncol(mm)){
        id = mm[x,y]
        bb = c(bb,id)
        
      }
    }
    bb = data.frame(bb)
    train_data = bind_cols(train_data,bb)
  }
  colnames(train_data) = c("id",bands)
  
  
  #### extract field and label
  fid = c()
  label = c()
  for(x in 1:nrow(lab_mat)){
    for(y in 1:ncol(lab_mat)){
      id = f_mat[x,y]
      lab= lab_mat[x,y]
      fid = c(fid,id)
      label = c(label,lab)
    }
  }

  
  dd = data.frame(folder = train_path[i,1],fid = fid,label = label,train_data) %>% filter(!is.na(label))  
  df = rbind(df,dd)
  rm(f_mat,lab_mat,fid,label,mm,bb,train_data);invisible(gc())
}

####################
ngb = data.frame()
for(i in 1:nrow(train_path)){
  cat("train_data_pca",i,"\n")
  f_mat = raster(train_path[i,2]) 
  f_mat = rasterToPoints(f_mat,spatial = T)
  llprj <-  "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
  llpts <- spTransform(f_mat, CRS(llprj))
  f_mat = as.data.frame(llpts)
  
  f_mat = f_mat %>% group_by(field_ids) %>% summarise_all(list(mean = mean)) %>% ungroup()
  
  ngb = rbind(ngb,f_mat)
  rm(f_mat);invisible(gc())
}

####################
field_details = data.frame()
for(i in 1:nrow(train_path)){
  cat("train_data",i,"\n")
  
  stac = jsonlite::read_json(paste0(data.dir,"/",label_path,"/",label_path,"_",train_path[i,1],"/stac.json"))
  bbox = unlist(stac$bbox)
  
  # Update tile coordinates
  tile_width = bbox [3] - bbox[1]
  tile_height = bbox [4] - bbox[2]
  
  dd = data.frame(folder = train_path[i,1],tile_width = tile_width,tile_height=tile_height)
  field_details = rbind(field_details,dd)
}


###################################################
####### GET TRAIN VEGETATION INDICIES AREA
###################################################
train_area_data = data.frame()
for(i in 1:nrow(train_path)){
  
  img  = c(paste0(data.dir,"/",image_path,"/",image_path,"_",train_path[i,1], "/B02.tif"),
           paste0(data.dir,"/",image_path,"/",image_path,"_",train_path[i,1],"/B03.tif"),
           paste0(data.dir,"/",image_path,"/",image_path,"_",train_path[i,1],"/B04.tif"),
           paste0(data.dir,"/",image_path,"/",image_path,"_",train_path[i,1],"/B08.tif"))
  mm2 = raster::stack(img)
  veg = c("EVI","SI","GLI","HUE","NDVI","GNDVI")
  area = c()
  for(j in 1:length(veg)){
    RemSoil <- FIELDimageR::fieldMask(mosaic = mm2, Red = 3, Green = 2, Blue = 1,NIR = 4, index = veg[j],
                                      plot = F,cropAbove = T)
    EX1.Canopy<-FIELDimageR::fieldArea(mosaic = RemSoil$mask,n.core = 16, plot = F)
    area = c(area,EX1.Canopy$areaPorcent$objArea)
  }
  pp = matrix(area,nrow = 1,ncol = length(veg))
  pp = cbind(folder = train_path[i,1],pp)
  train_area_data = rbind(train_area_data,pp)
}
colnames(train_area_data) = c("folder",paste0("Area_",veg))

for(i in 2:ncol(train_area_data)){
  train_area_data[,i] = as.numeric(train_area_data[,i])
}
 
field_details = field_details %>% left_join(train_area_data) %>% left_join(df %>% dplyr::select(folder,fid))





##########################################
### GETTING TEST DATASET 4 HOURS
##########################################
### Get the test label json file
colls =  paste0(data.dir,"/",test_path,"/",test_path,"/collection.json")
colls = jsonlite::read_json(colls)
qq = colls$links[3:length(colls$links)-1]
ids = c()
field_path = c()
labels_path = c()
for(i in 1:length(qq)){
  id = map.func(qq[[i]]$href) %>% as.character()
  id_path = paste0(data.dir,"/",test_path,"/",test_path,"/",test_path,"_",id,"/field_ids.tif")
  ids = c(ids,id)
  field_path = c(field_path,id_path)
}

test_fields = data.frame(ids = ids, field_path = field_path) %>% filter(!is.na(ids))


test = data.frame()
for(i in 1:nrow(test_fields)){
  cat("test_band_data",i,"\n")
  f_mat = raster(test_fields[i,2]) %>% as.matrix()
  f_mat_rs = which(rowSums(f_mat,na.rm = T)>0)
  f_mat_cs = which(colSums(f_mat,na.rm = T)>0)
  f_mat = as.matrix(f_mat[f_mat_rs,f_mat_cs])

  #### extract bands
  bands = c(paste0("B0",1:9),"B11","B12")
  train_data = data.frame(id= 1)
  for(b in bands){
    img = paste0(data.dir,"/",image_path,"/",image_path,"_",test_fields[i,1],"/",b,".tif")
    mm = raster(img) %>% as.matrix()
    mm = as.matrix(mm[f_mat_rs,f_mat_cs])


    bb = c()
    for(x in 1:nrow(mm)){
      for(y in 1:ncol(mm)){
        id = mm[x,y]
        bb = c(bb,id)

      }
    }
    bb = data.frame(bb)
    train_data = bind_cols(train_data,bb)
  }
  colnames(train_data) = c("id",bands)

  fid = c()
  for(x in 1:nrow(f_mat)){
    for(y in 1:ncol(f_mat)){
      id = f_mat[x,y]
      fid = c(fid,id)
    }
  }


  dd = data.frame(folder = test_fields[i,1],fid = fid,train_data) %>% filter(!is.na(fid))
  test = rbind(test,dd)
  rm(f_mat,fid,dd,bb,train_data,f_mat_cs,fmat_rs);invisible(gc())
}


test_ngb = data.frame()
for(i in 1:nrow(test_fields)){
  cat("test_data",i,"\n")
  f_mat = raster(test_fields[i,2]) 
  f_mat = rasterToPoints(f_mat,spatial = T)
  llprj <-  "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0"
  llpts <- spTransform(f_mat, CRS(llprj))
  f_mat = as.data.frame(llpts)
  
  f_mat = f_mat %>% group_by(field_ids) %>% summarise_all(list(mean = mean)) %>% ungroup()
  uniq_field = unique(f_mat$field_ids)
  
  
  test_ngb = rbind(test_ngb,f_mat)
  rm(f_mat);invisible(gc())
}


####################
test_details = data.frame()
for(i in 1:nrow(test_fields)){
  cat("test_data_tile",i,"\n")
  
  img = jsonlite::read_json(paste0(data.dir,"/",test_path,"/",test_path,"/",test_path,"_",test_fields[i,1],"/stac.json"))
  bbox = unlist(img$bbox)
  
  # Update tile coordinates
  tile_width = bbox [3] - bbox[1]
  tile_height = bbox [4] - bbox[2]
  

  dd = data.frame(folder = test_fields[i,1],tile_width = tile_width,tile_height=tile_height)
  test_details = rbind(test_details,dd)
}


###################################################
####### GET TRAIN VEGETATION INDICIES AREA
###################################################
test_area_data = data.frame()
for(i in 1:nrow(test_fields)){
  cat("test_area_data",i,"\n")
  img  = c(paste0(data.dir,"/",image_path,"/",image_path,"_",test_fields[i,1], "/B02.tif"),
           paste0(data.dir,"/",image_path,"/",image_path,"_",test_fields[i,1],"/B03.tif"),
           paste0(data.dir,"/",image_path,"/",image_path,"_",test_fields[i,1],"/B04.tif"),
           paste0(data.dir,"/",image_path,"/",image_path,"_",test_fields[i,1],"/B08.tif"))
  mm2 = raster::stack(img)
  veg = c("EVI","SI","GLI","HUE","NDVI","GNDVI")
  area = c()
  for(j in 1:length(veg)){
    RemSoil <- FIELDimageR::fieldMask(mosaic = mm2, Red = 3, Green = 2, Blue = 1,NIR = 4, index = veg[j],
                                      plot = F,cropAbove = T)
    EX1.Canopy<-FIELDimageR::fieldArea(mosaic = RemSoil$mask,n.core = 16, plot = F)
    area = c(area,EX1.Canopy$areaPorcent$objArea)
  }
  pp = matrix(area,nrow = 1,ncol = length(veg))
  pp = cbind(folder = test_fields[i,1],pp)
  test_area_data = rbind(test_area_data,pp)
}
colnames(test_area_data) = c("folder",paste0("Area_",veg))


for(i in 2:ncol(test_area_data)){
  test_area_data[,i] = as.numeric(test_area_data[,i])
}

test_details = test_details %>% 
  left_join(test_area_data) %>% 
  left_join(test %>% dplyr::select(folder,fid))




#############################################################
#### PREPARE FINAL DATA FOR TRAINING
###############################################################
neighbour_data = rbind(ngb,test_ngb)
dr.dat = cbind(x_mean=neighbour_data$x_mean,y_mean=neighbour_data$y_mean) %>% as.data.frame()
transform = preProcess(dr.dat, method = c("center","scale","pca"))
pc = predict(transform, dr.dat) %>% as.data.frame()
neighbour_data$y_mean_pca = pc$PC1
neighbour_data$x_mean_pca = pc$PC2

###########
transform = preProcess(dr.dat, method = c("center","scale"))
pc = predict(transform, dr.dat) %>% as.data.frame()
neighbour_data$y_mean_scale = pc$y_mean
neighbour_data$x_mean_scale = pc$x_mean

neighbour_data$lat1 = cos(neighbour_data$y_mean_scale)*cos(neighbour_data$x_mean_scale)
neighbour_data$lat2 = cos(neighbour_data$y_mean_scale)*sin(neighbour_data$x_mean_scale)
neighbour_data$lat3 = sin(neighbour_data$y_mean_scale)

neighbour_data$rot45_x = 0.707 * neighbour_data$y_mean_scale + 0.707*neighbour_data$x_mean_scale
neighbour_data$rot45_y = 0.707 * neighbour_data$x_mean_scale - 0.707*neighbour_data$y_mean_scale
neighbour_data$rot30_x = 0.866 * neighbour_data$y_mean_scale + 0.5*neighbour_data$x_mean_scale
neighbour_data$rot30_y = 0.866 * neighbour_data$x_mean_scale - 0.5*neighbour_data$y_mean_scale
neighbour_data$rot60_x = 0.5 * neighbour_data$y_mean_scale + 0.866*neighbour_data$x_mean_scale
neighbour_data$rot60_y = 0.5 * neighbour_data$x_mean_scale - 0.866*neighbour_data$y_mean_scale


neighbour_data$x_mean2 = as.numeric(deg2dms(neighbour_data$x_mean)[,2])*60
neighbour_data$y_mean2 = as.numeric(deg2dms(neighbour_data$y_mean)[,2])*60

neighbour_data$x_mean = NULL
neighbour_data$y_mean= NULL


############# #####################################
######  TRAIN DATA FEATURE ENGINEERING
###################################################
df = df  %>% 
  mutate(
    ndvi =(B08 - B04)/ (B08 + B04),
    GLI = 	(2*B03-B04-B02)/(2*B03+B04+B02),
    CVI = (B08 / B03) * (B04 / B03),
    SIPI = (B08 - B02) / (B08 - B04),
    S2REP = 705 + 35 * ((((B07 + B04)/2) - B05)/(B06 - B05)),
    CCCI = ((B08 - B05) / (B08 + B05)) / ((B08 - B04) / (B08 + B04)),
    hue = atan(2*(B02-B03-B04)/30.5*(B03-B04)),
    RENDVI = (B06 - B05) / (B06 + B05), 
    RECI = (B08 / B04)-1,
    RECI2 = (B08 / B05)-1,
    evi = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0),
    evi2 = 2.4 * (B08 - B04) / (B08 + B04 + 1.0),
    npcri = (B04 - B02) / (B04 + B02),
    ndwi = (B03 - B08) / (B03 + B08)
  )

####### CALCLUATE AGGREGATE FEATURE PER FIELD ID
df_train = df %>% filter(!is.na(fid)) %>% 
  group_by(fid) %>%
  mutate(field_tile_count = n(), field_overlap_count = length(unique(folder))) %>%  ungroup() %>% 
  dplyr::select(-c(folder,id)) %>% 
  group_by(fid) %>% 
  summarise_all(list(
    median =median,
    max = max)) %>% 
  ungroup() %>%
  dplyr::select(-c(label_max,field_tile_count_max,field_overlap_count_max))


###### EXTARCT FEATURE ENGINEERING - JOINING WITH OTHER DATA FILES
df_train = df_train %>% 
  left_join(field_details %>% dplyr::select(-folder) %>% filter(!duplicated(fid))) %>% 
  left_join(neighbour_data %>% filter(!duplicated(field_ids))%>% rename(fid = field_ids)) %>% 
  mutate(field_tile_size = 20000*field_tile_count_median*tile_width*tile_height,
         fid = NULL)

label = df_train$label_median
df_train$label_median=NULL



###############################################
#####  TEST DATA FEATURE ENGINEERING
###############################################
test = test %>% 
  mutate(
    ndvi =(B08 - B04)/ (B08 + B04),
    GLI = 	(2*B03-B04-B02)/(2*B03+B04+B02),
    CVI = (B08 / B03) * (B04 / B03),
    SIPI = (B08 - B02) / (B08 - B04),
    S2REP = 705 + 35 * ((((B07 + B04)/2) - B05)/(B06 - B05)),
    CCCI = ((B08 - B05) / (B08 + B05)) / ((B08 - B04) / (B08 + B04)),
    hue = atan(2*(B02-B03-B04)/30.5*(B03-B04)),
    RENDVI = (B06 - B05) / (B06 + B05), 
    RECI = (B08 / B04)-1,
    RECI2 = (B08 / B05)-1,
    evi = 2.5 * (B08 - B04) / ((B08 + 6.0 * B04 - 7.5 * B02) + 1.0),
    evi2 = 2.4 * (B08 - B04) / (B08 + B04 + 1.0),
    npcri = (B04 - B02) / (B04 + B02),
    ndwi = (B03 - B08) / (B03 + B08))

#################################  FINAL TEST DATA
df_test = test %>% filter(!is.na(fid)) %>% 
  group_by(fid) %>%
  mutate(field_tile_count = n(), field_overlap_count = length(unique(folder))) %>%  ungroup() %>% 
  dplyr::select(-c(folder,id))  %>% group_by(fid) %>% 
  summarise_all(list(median =median,
                     max = max)) %>%  ungroup() 

df_test = df_test %>% 
         left_join(test_details%>% dplyr::select(-folder) %>% filter(!duplicated(fid))) %>% 
         mutate(field_tile_size = 20000*field_tile_count_median*tile_width*tile_height) %>% 
          left_join(neighbour_data %>% filter(!duplicated(field_ids))%>% rename(fid = field_ids)) 


####### WRITE OUT TRAIN/TEST DATASET to DATA FOLDER
fwrite(df_test, file = paste0(output_path,"/Final_Test.csv"), row.names = F)