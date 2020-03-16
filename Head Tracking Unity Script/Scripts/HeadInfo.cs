using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

public class HeadInfo : MonoBehaviour
{
    public GameObject Head;
    public GameObject Left;
    public GameObject Right;
    public float xPos;
    public float yPos;
    public float zPos;
    public float xAng;
    public float yAng;
    public float zAng;

    public float xLeftPos;
    public float yLeftPos;
    public float zLeftPos;
    public float xLeftAng;
    public float yLeftAng;
    public float zLeftAng;

    public float xRightPos;
    public float yRightPos;
    public float zRightPos;
    public float xRightAng;
    public float yRightAng;
    public float zRightAng;
    public GameObject canvas;
    public string filePath;





    void Start()
    {
        string path = Application.persistentDataPath + "/data.csv";
        filePath = path;

        Debug.Log("filePath: " + filePath);
        string delimiter = ",";

        if (File.Exists(filePath))
            File.Delete(filePath);


        StringBuilder sb = new StringBuilder();

        sb.AppendLine("TimeStamp" + delimiter +
                      "head_x" + delimiter + "head_y" + delimiter + "head_z" + delimiter +
                      "head_pitch" + delimiter + "head_yaw" + delimiter + "head_roll" + delimiter +
                      "left_c_x" + delimiter + "left_c_y" + delimiter + "left_c_z" + delimiter +
                      "left_c_pitch" + delimiter + "left_c_yaw" + delimiter + "left_c_roll" + delimiter +
                      "right_c_x" + delimiter + "right_c_y" + delimiter + "right_c_z" + delimiter +
                      "right_c_pitch" + delimiter + "right_c_yaw" + delimiter + "right_c_roll" + "\n");

        // x --> Pitch
        // y --> Yaw
        // z --> Roll




        if (!File.Exists(filePath))
            File.WriteAllText(filePath, sb.ToString());
        else
            File.AppendAllText(filePath, sb.ToString());

    }



    // Update is called once per frame
    void Update()

    {



        xPos = Head.transform.position.x;
        yPos = Head.transform.position.y;
        zPos = Head.transform.position.z;

        xAng = Head.transform.eulerAngles.x;
        yAng = Head.transform.eulerAngles.y;
        zAng = Head.transform.eulerAngles.z;

        xRightPos = Right.transform.position.x;
        yRightPos = Right.transform.position.y;
        zRightPos = Right.transform.position.z;

        xRightAng = Right.transform.eulerAngles.x;
        yRightAng = Right.transform.eulerAngles.y;
        zRightAng = Right.transform.eulerAngles.z;

        xLeftPos = Left.transform.position.x;
        yLeftPos = Left.transform.position.y;
        zLeftPos = Left.transform.position.z;

        xLeftAng = Left.transform.eulerAngles.x;
        yLeftAng = Left.transform.eulerAngles.y;
        zLeftAng = Left.transform.eulerAngles.z;

/*
        Text txt = canvas.GetComponent<Text>();
        txt.text = "File path " + filePath + "\n" +
                   " xPos " + xPos + " yPos " + yPos + " zPos " + zPos + "\n" +
                   " xAng " + xAng + " yAng " + yAng + " zAng" + zAng + "\n" +
                   " xRightPos " + xRightPos + " yRightPos " + yRightPos + " zRightPos " + zRightPos + "\n" +
                   " xRightAng " + xRightAng + " yRightAng " + yRightAng + " zRightAng" + zRightAng + "\n" +
                   " xLeftPos " + xLeftPos + " yLeftPos " + yLeftPos + " zLeftPos " + zLeftPos + "\n" +
                   " xLeftAng " + xLeftAng + " yLeftAng " + yLeftAng + " zLeftAng" + zLeftAng + "\n";
                   */


    //----------------------------------------------------------------------------------------------------------------//
        //WRITE THE CSV

        
        Debug.Log("filePath: " + filePath);

        string delimiter = ",";

        StringBuilder sb = new StringBuilder();

        sb.AppendLine(DateTime.Now.ToUniversalTime() + delimiter +
                      xPos + delimiter + yPos + delimiter + zPos + delimiter +
                      xAng + delimiter + yAng + delimiter + zAng + delimiter +
                      xLeftPos + delimiter + yLeftPos + delimiter + zLeftPos + delimiter +
                      xLeftAng + delimiter + yLeftAng + delimiter + zLeftAng + delimiter +
                      xRightPos + delimiter + yRightPos + delimiter + zRightPos + delimiter +
                      xRightAng + delimiter + yRightAng + delimiter + zRightAng);

        if (!File.Exists(filePath))
            File.WriteAllText(filePath, sb.ToString());
        else
            File.AppendAllText(filePath, sb.ToString());

    //----------------------------------------------------------------------------------------------------------------//

    }




}
