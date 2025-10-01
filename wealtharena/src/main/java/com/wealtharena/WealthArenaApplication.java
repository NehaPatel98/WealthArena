package com.wealtharena;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
public class WealthArenaApplication {

	public static void main(String[] args) {
		SpringApplication.run(WealthArenaApplication.class, args);
	}
}


