import ee
import random
from time import sleep

ee.Authenticate()
ee.Initialize(project='giel-douglas-thesis')

print("\nAuthenticated\n")

res = 60 # resolution of obtained images (if change, should change region size to maintain 512x512)

grid_list = []

# (from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)
def applyScaleFactors(image):
  opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
  return image.addBands(opticalBands, names=None, overwrite=True)

def export(year: str, num_images: int, thresholds: list, max_cloud_coverage: int):

    # specify region
    amazon = ee.FeatureCollection('projects/giel-douglas-thesis/assets/brazilian_legal_amazon').geometry() 

    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(amazon).filterMetadata("CLOUD_COVER", "LESS_THAN", max_cloud_coverage).select(['SR_B4', 'SR_B3', 'SR_B2']).filterDate(year+"-07-01", year+"-9-01")

    
    landsat = landsat.map(applyScaleFactors)

    # get mosaic of reduced resolution landsat images 
    landsat_geo = landsat.geometry() # get geometry of landsat region
    landsat_reduced = landsat.mosaic().reproject(landsat_geo.projection(), None, res)


    # get Hansen image series
    hansen =  ee.Image('UMD/hansen/global_forest_change_2022_v1_10')
    

    # get bands
    treecover2000 = hansen.select(["treecover2000"])
    lossyear = hansen.select(["lossyear"])
    loss = hansen.select(["loss"])
    gain = hansen.select(["gain"])
  
    treecover_reduced_list = []
    for threshold in thresholds:
        # mask the treecover image based on threshold
        treecover2000_mask = treecover2000.where(treecover2000.lt(threshold), 0)
        treecover2000_mask = treecover2000_mask.where(treecover2000_mask.gte(threshold), 1)

        
        
        # consider loss between 2000-2012
        # t = ee.Image(loss.eq(1)).and(lossyear.lte(12))
        treecover = treecover2000_mask.where(loss.eq(1).bitwiseAnd(lossyear.lte(12)), 0)

        # consider gain between 2000-2012
        treecover = treecover.where(gain.eq(1), 1)

        # consider loss between 2012-given year
        treecover = treecover.where(loss.eq(1).bitwiseAnd(lossyear.gt(12)), 0)

        # reduce resolution
        treecover_reduced_list.append(treecover.reproject(treecover.projection(), None, scale=res))


    
    # create grid partition over mosaic
    grid_scale = 30700 # gives 512x512 image for image of 60m res

    grid = landsat_geo.coveringGrid(proj=landsat_geo.projection(), scale=grid_scale)
    grid_list =  grid.toList(grid.size())


    # generate random list of grids
    num_grids = grid.size().getInfo()


    grid_nums_shuffled = list(range(num_grids))
    random.shuffle(grid_nums_shuffled)


    # select grids that are valid (fully covered by amazon region with less than max cloud coverage)
    grid_nums = []
    selected_grids = []
    centers = []

    counter = 0
    idx = 0
    while(counter<num_images):

        # loop until find a grid space that is fully covered by landsat geo (ensures we have image that covers grid slot)
        # TODO: maybe filter out grid collection before entering loop
        while(not landsat_geo.contains(ee.Feature(grid_list.get(grid_nums_shuffled[idx])).geometry(), .001).getInfo()):
            print("Grid #", idx, "is invalid")
            idx+=1
            if(idx>=num_grids):
                print("ONLY "+str(counter)+" SELECTED. REMAINING ARE NOT COVERED")
                num_images = counter # TODO: check if valid
                break
        if(counter<num_images): # check if valid grid found in previous loop (if num_images was reduced)
            print("Grid #", idx, " is valid")
            grid_nums.append(grid_nums_shuffled[idx])
            selected_grids.append(ee.Feature(grid_list.get(grid_nums_shuffled[idx])).geometry())
            centers.append(selected_grids[-1].centroid(.001).getInfo().get('coordinates'))
        counter +=1
        idx+=1

    

    # get corresponding Landsat and Hansen images
    selected_landsat_images = []
    selected_treecover_images = [[] for i in range(len(thresholds))]
    for i in range(num_images):
        current_partition = selected_grids[i]
        selected_landsat_images.append(landsat_reduced.clip(current_partition))

        # get hansen images for each threshold
        for j in range(len(thresholds)):
            selected_treecover_images[j].append(treecover_reduced_list[j].clip(current_partition))



    # Export images:
    
    # landsat folder name depends on visualization type
    landsat_folder = "landsat_images"+'_year:'+year+'_res:'+str(res)+'_vis:ee'
    
    for i in range(num_images):

        # get visualizations of images
        
        # (from https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2)
        rgb_image_vis = selected_landsat_images[i].visualize(
            bands = ['SR_B4', 'SR_B3', 'SR_B2'],
            max = .3, 
            min = 0
        )
        
        export1 = ee.batch.Export.image.toDrive(
            scale = res,
            image = rgb_image_vis,
            description = 'landsat_image_'+'{:03d}'.format(i)+'_year:'+year+'_res:'+str(res)+'_lon:'+str(centers[i][0])+'_lat:'+str(centers[i][1]),
            fileFormat = 'GeoTIFF',
            folder = landsat_folder,
            crs='EPSG:3857'
        )
        export1.start()

        export2_list = []
        for j in range(len(thresholds)):
            treecover_vis = selected_treecover_images[j][i].visualize(
                    max = 1,
                    min = 0,
                    palette = ['black', 'white']
            )

            export2 = ee.batch.Export.image.toDrive(
                scale = res,
                image = treecover_vis,
                description = 'hansen_image_'+'{:03d}'.format(i)+'_year:'+year+'_res:'+str(res)+'_thresh:'+str(thresholds[j])+'_lon:'+str(centers[i][0])+'_lat:'+str(centers[i][1]),
                fileFormat = 'GeoTIFF',
                folder = "hansen_images"+'_year:'+year+'_res:'+str(res)+'_thresh:'+str(thresholds[j]),
                crs='EPSG:3857'

            )
            export2.start()
            export2_list.append(export2)


        print("\nExport #", i, ": \n")

        # check status of exports

        task_id_1 = export1.status().get('id')

        task_id_2_list = []
        for i in range(len(thresholds)):
            task_id_2_list.append(export2.status().get('id'))


        if(task_id_1 and task_id_2_list[0]):
            status1 = ee.data.getTaskStatus(task_id_1)[0].get('state')
            status2 = ee.data.getTaskStatus(task_id_2_list[-1])[0].get('state')
            while(status2!= 'COMPLETED' or status1!='COMPLETED'):
                status1 = ee.data.getTaskStatus(task_id_1)[0].get('state')
                status2 = ee.data.getTaskStatus(task_id_2_list[0])[0].get('state')
                print("export 1:", status1)
                print("export 2:", status2)
                sleep(5)
        elif(task_id_1):
            print("Export 2 not initiated")
        else:
            print("Export 1 not intiated")

    print("DONE (year: "+year+", num_images: "+str(num_images)+", threshold: "+str(threshold)+", max_cloud_coverage: "+str(max_cloud_coverage)+")")




export("2022", 20, [80], 10)