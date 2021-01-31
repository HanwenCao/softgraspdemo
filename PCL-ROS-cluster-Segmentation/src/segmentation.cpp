/*

ROS node for point cloud cluster based segmentaion of cluttered objects on table

Author: Sean Cassero
7/15/15

*/


#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <obj_recognition/SegmentedClustersArray.h>
//#include <obj_recognition/ClusterData.h>
#include <pcl/filters/crop_box.h>
#include<cmath>

class segmentation {

public:

  explicit segmentation(ros::NodeHandle nh) : m_nh(nh)  {

    // define the subscriber and publisher

    //m_sub = m_nh.subscribe ("/obj_recognition/point_cloud", 1, &segmentation::cloud_cb, this);
    m_clusterPub = m_nh.advertise<obj_recognition::SegmentedClustersArray> ("obj_recognition/pcl_clusters",1);
    m_sub = m_nh.subscribe ("/camera/depth/color/points", 1, &segmentation::cloud_cb, this);
    m_pub = m_nh.advertise<pcl::PCLPointCloud2> ("vis_intermidiate",1);
    m_pub2 = m_nh.advertise<sensor_msgs::PointCloud2> ("vis_result",1);

  }

private:

ros::NodeHandle m_nh;
ros::Publisher m_pub, m_pub2;
ros::Subscriber m_sub;
ros::Publisher m_clusterPub;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);

float dist_to_plane(float a,float b, float c, float d,float x,float y,float z){
    // distance of a point(x,y,z) to a plane(ax+by+cz+d=0)
    float dist1 = std::fabs(a*x+b*y+c*z+d);
    float dist2 = std::sqrt(a*a+b*b+c*c);
    float dist = dist1/dist2;
    return dist;
}

}; // end class definition



// define callback function
void segmentation::cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{



  // Container for original & filtered data
  pcl::PCLPointCloud2* cloud = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
  pcl::PCLPointCloud2* cloud_filtered = new pcl::PCLPointCloud2;
  pcl::PCLPointCloud2Ptr cloudFilteredPtr (cloud_filtered);


  // Convert to PCL data type
  pcl_conversions::toPCL(*cloud_msg, *cloud);


  // Perform voxel grid downsampling filtering
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloudPtr);
  sor.setLeafSize (0.01, 0.01, 0.01);
  sor.filter (*cloudFilteredPtr);
  // publish the voxel (pcl::PCLPointCloud2 tpye)
  //m_Pub.publish(*cloudFilteredPtr);


  pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud = new pcl::PointCloud<pcl::PointXYZRGB>;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtr (xyz_cloud); // need a boost shared pointer for pcl function inputs

  // convert the pcl::PointCloud2 tpye to pcl::PointCloud<pcl::PointXYZRGB>
  pcl::fromPCLPointCloud2(*cloudFilteredPtr, *xyzCloudPtr);


  // write the whole scene to disk
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZRGB> ("scene.pcd", *xyzCloudPtr, false);


  //perform passthrough filtering to remove table leg

  // create a pcl object to hold the passthrough filtered results
  pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrFiltered (xyz_cloud_filtered);

  // Create the filtering object
  pcl::PassThrough<pcl::PointXYZRGB> pass;
  pass.setInputCloud (xyzCloudPtr);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (-2, 2);  //(.5, 1.1);
  //pass.setFilterLimitsNegative (true);
  pass.filter (*xyzCloudPtrFiltered);

  // further filtering with a 3D bbox
  pcl::CropBox<pcl::PointXYZRGB> boxFilter;
  boxFilter.setMin(Eigen::Vector4f(-2, -2, -2, 1.0));
  boxFilter.setMax(Eigen::Vector4f(2, 2, 2, 1.0));
  boxFilter.setInputCloud(xyzCloudPtr);
  boxFilter.filter(*xyzCloudPtrFiltered);

  // convert to pcl::PCLPointCloud2
  pcl::PCLPointCloud2 outputPCL0;
  pcl::toPCLPointCloud2( *xyzCloudPtrFiltered ,outputPCL0);
  // publish the pass through (pcl::PCLPointCloud2 tpye)
  //m_pub.publish(outputPCL0);


  // create a pcl object to hold the ransac filtered results
  pcl::PointCloud<pcl::PointXYZRGB> *xyz_cloud_ransac_filtered = new pcl::PointCloud<pcl::PointXYZRGB>;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrRansacFiltered (xyz_cloud_ransac_filtered);


  // perform ransac planar filtration to remove table top
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<pcl::PointXYZRGB> seg1;
  // Optional
  seg1.setOptimizeCoefficients (true);
  // Mandatory
  seg1.setModelType (pcl::SACMODEL_PLANE);
  seg1.setMethodType (pcl::SAC_RANSAC);
  seg1.setDistanceThreshold (0.01); //0.04

  seg1.setInputCloud (xyzCloudPtrFiltered);
  seg1.segment (*inliers, *coefficients); //inliers: points belong to the plane

  //output the plane coefficients (in ax+by+cz+d=0 form)
  std::cerr << "\nPlane coefficients: " << coefficients->values[0] << " " 
                                  << coefficients->values[1] << " "
                                  << coefficients->values[2] << " " 
                                  << coefficients->values[3] << std::endl;
  float plane_a = coefficients->values[0];float plane_b = coefficients->values[1];float plane_c = coefficients->values[2];float plane_d = coefficients->values[3];



  // Create the filtering object
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  //extract.setInputCloud (xyzCloudPtrFiltered);
  extract.setInputCloud (xyzCloudPtrFiltered);
  extract.setIndices (inliers);
  extract.setNegative (true);
  extract.filter (*xyzCloudPtrRansacFiltered);

  // convert to pcl::PCLPointCloud2
  pcl::toPCLPointCloud2( *xyzCloudPtrRansacFiltered ,outputPCL0);
  // publish point clouds with table removed (pcl::PCLPointCloud2 type)
  m_pub.publish(outputPCL0);



  // perform euclidean cluster segmentation to seporate individual objects

  // Create the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (xyzCloudPtrRansacFiltered);

  // create the extraction object for the clusters
  // cluster_indices[0]: PointIndices for the first cluster
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  // specify euclidean cluster parameters
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (xyzCloudPtrRansacFiltered);
  // exctract the indices pertaining to each cluster and store in a vector of pcl::PointIndices
  ec.extract (cluster_indices);
  std::cout<<"Found "<<cluster_indices.size()<<" objects."<<std::endl;

  // declare an instance of the SegmentedClustersArray message
  obj_recognition::SegmentedClustersArray CloudClusters;

  // declare the output variable instances
  sensor_msgs::PointCloud2 output;
  pcl::PCLPointCloud2 outputPCL;

  // here, cluster_indices is a vector of indices for each cluster. iterate through each indices object to work with them seporately
  int i_cluster = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) //for each object/cluster
  {
    i_cluster++;
    // create a new clusterData message object
    //obj_recognition::ClusterData clusterData;

    // create a pcl object to hold the extracted cluster (point cloud of this object/cluster)
    pcl::PointCloud<pcl::PointXYZRGB> *cluster = new pcl::PointCloud<pcl::PointXYZRGB>;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clusterPtr (cluster);  //

    // now we are in a vector of indices pertaining to a single cluster.
    // Assign each point corresponding to this cluster in xyzCloudPtrPassthroughFiltered a specific 'color' for identification purposes
    std::cout<<"  "<<it->indices.size()<<"  points."<<std::endl;  //num of points belongs to this object/cluster
    float dist_sum=0;
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      // store points belong to this cluster to a cloud "clusterPtr"
      clusterPtr->points.push_back(xyzCloudPtrRansacFiltered->points[*pit]);
      
      //distance of points to the plane (do it for each object)
      //and estimate the volumn of the object by sum up the distances
      //Pay attention that
      //this method only work for estimating the volume of a cloud when the camera is shotting downward
      //Otherwise, say if there is a vertical surface of an object that is captured by RGB-D camera, 
      //many points belongs to that surface will make the estimated volume much larger than it should be.
      float x = xyzCloudPtrRansacFiltered->points[*pit].x;
      float y = xyzCloudPtrRansacFiltered->points[*pit].y;
      float z = xyzCloudPtrRansacFiltered->points[*pit].z;
      float dist = dist_to_plane(plane_a,plane_b,plane_c,plane_d,x,y,z);
      dist_sum += dist;
    }
    std::cout<<" Approxiamated volume of this object/cluster: "<<dist_sum<<std::endl;

    

    // log the position of the cluster
    //clusterData.position[0] = (*cloudPtr).data[0];
    //clusterData.position[1] = (*cloudPtr).points.back().y;
    //clusterData.position[2] = (*cloudPtr).points.back().z;
    //std::string info_string = string(cloudPtr->points.back().x);
    //printf(clusterData.position[0]);


    // Write the cloud of this cluster to disk
    clusterPtr->width = 1;
    clusterPtr->height = clusterPtr->points.size();
    pcl::PCDWriter writer;
    std::stringstream ss;
    ss << "cluster_" << i_cluster << ".pcd";
    writer.write<pcl::PointXYZRGB> (ss.str (), *clusterPtr, false);


    // convert from pcl::PointCloud<pcl::PointXYZRGB> to pcl::PCLPointCloud2
    pcl::toPCLPointCloud2( *clusterPtr ,outputPCL);
    //m_pub.publish(outputPCL);  //For frame []: Frame [] does not exist ??

    // Convert from pcl::PCLPointCloud2 to ROS data type
    pcl_conversions::fromPCL(outputPCL, output);

    // add the cluster to the array message
    //clusterData.cluster = output;
    CloudClusters.clusters.push_back(output);
    

  }

  // publish the clusters
  m_clusterPub.publish(CloudClusters);
  //m_pub2.publish(output);



}



int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "segmentation");
  ros::NodeHandle nh;

  segmentation segs(nh);

  while(ros::ok())
  ros::spin ();

}
